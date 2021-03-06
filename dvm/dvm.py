"""
This module implements the Discrete Voter Model for ecological inference in
Python 3.
"""

import time
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm.autonotebook import trange

import expec_votes as ev
import phc
import prob_votes as pv
import tools

# Suppress TenorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def chain_mle(chain_results):
    """
    Find the Maximum Likelihood Estimate (MLE) of the distribution of
    PHCs indicated by the sample generated by HMC or RWM.

    chain_results (dict): Python dictionary containing the sample, the
    type of scorer, and traces of log probability and log acceptance

    return: the PHC with the maximum likelihood and its index
    """
    index = tf.math.argmax(chain_results['log_prob_trace'])
    return tools.prob_normalize(chain_results['sample'][index]), chain_results['log_prob_trace'][index]


def mean_phc(chain_results):
    """
    Find the mean of the distribution of PHCs indicatedby the sample
    generated by HMC or RWM.

    chain_results (dict): Python dictionary containing the sample, the
    type of scorer, and traces of log probability and log acceptance

    return: the mean PHC
    """
    normalized_results = tf.map_fn(
        tools.prob_normalize, chain_results['sample'])
    return tf.math.reduce_mean(normalized_results, axis=0)


def rwm_proposal_fn(states, seed, epsilon=0.001):
    """
    Generate a proposal for the Random Walk Metropolis kernel.

    states (Tensor): the current state of the Markov chain, as a list
    seed (int): the seed used to generate the proposal
    epsilon (float): the value to add and subtract to places in the PHC
    """
    # Extract the single state from the `states` list
    state = states.pop()

    # Select two random cells in the PHC
    size = tf.reduce_prod(state.shape)
    cell_1_index = tf.random.uniform(
        shape=[], minval=0, maxval=size - 1, dtype=tf.int32)
    cell_2_index = tf.random.uniform(
        shape=[], minval=0, maxval=size - 1, dtype=tf.int32)

    # Flatten the state and change the cells
    flattened_state = tf.reshape(state, [-1]).numpy()
    flattened_state[cell_1_index] -= epsilon
    flattened_state[cell_2_index] += epsilon

    # Check if either cell has dipped below 0
    # If so, "reject" by returning the previous state
    if flattened_state[cell_1_index] < 0 or flattened_state[cell_2_index] < 0:
        return [state]

    # Convert back to a Tensor and return the PHC
    tf_state = tf.convert_to_tensor(flattened_state, dtype=state.dtype)

    return [tf.reshape(tf_state, state.shape)]


def init_hmc_kernel(log_prob_fn, step_size, num_adaptation_steps=0):
    """
    Initialize the HMC kernel.

    log_prob_fn (function): the function to calculate the log probability
    step_size (float): the float size to use for the kernel
    num_adaptation_steps (int): the number of adaptation steps

    return: kernel
    """
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=3)
    return tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_kernel,
        num_adaptation_steps=num_adaptation_steps)


def init_rwm_kernel(log_prob_fn):
    """
    Initialize the RWM kernel.

    log_prob_fn (function): the function to calculate the log probability

    return: kernel
    """
    return tfp.mcmc.RandomWalkMetropolis(log_prob_fn, new_state_fn=rwm_proposal_fn)


def hmc_trace_fn(_, pkr):
    return pkr.inner_results.accepted_results.target_log_prob, pkr.inner_results.log_accept_ratio, pkr.inner_results.accepted_results.step_size


def rwm_trace_fn(_, pkr):
    return pkr.accepted_results.target_log_prob, pkr.log_accept_ratio


def sample_chain(kernel, n_iter, current_state, trace_fn=None):
    return tfp.mcmc.sample_chain(
        num_results=n_iter,
        num_steps_between_results=2,
        current_state=current_state,
        kernel=kernel,
        trace_fn=trace_fn)


def burn_in(chain_result_tensor, burn_frac):
    num_samples = chain_result_tensor.shape[0]
    start = int(burn_frac * num_samples)

    begin = [start]
    size = [num_samples - start]

    for dim in chain_result_tensor.shape[1:]:
        begin.append(0)
        size.append(dim)

    return tf.slice(chain_result_tensor, begin, size)


def dvm_elections(election, candidate=None, phc_granularity=10, use_hmc=False,
                  expec_scoring=False, burn_frac=0.3, n_steps=200, n_iter=1,
                  verbose=False):
    """
    Run the Discrete Voter Model on an Election.

    election (Election): the election to analyze
    candidate (string): the candidate to analyze
    phc_granularity (int): the size of a dimension of the PHC
    use_hmc (bool): whether to use the HMC or RWM kernel
    expec_scoring (bool): whether to score by:
        1. the probability of a PHC to produce the outcome
        (False, default)
        2. the difference in the outcome and the PHC's expectation
        (True)
    burn_frac (float): the fraction of MCMC iterations to burn
    n_steps (int): the number of steps to run the MCMC for
    verbose (bool): whether to display loogging and progress bars

    return: a list of dictionaries of the election name, chain results and time
    """
    # Create an initial grid
    initial_phc = phc.make_phc(election.num_demo_groups, phc_granularity)

    # Get the observed votes for the desired candidate
    if not candidate:
        candidate = election.candidates[0]

    cand_obs_votes = {}
    for prec in election.precincts:
        cand_obs_votes[prec] = election.vote_totals[prec][candidate]

    # Run the MCMC with the specified kernel
    total_time = 0
    total_time -= time.time()

    if use_hmc:
        chain_results = hmc(n_steps, burn_frac, initial_phc,
                            election.dpp, cand_obs_votes,
                            expec_scoring=expec_scoring,
                            verbose=verbose)
    else:
        chain_results = rwm(n_steps, burn_frac, initial_phc,
                            election.dpp, cand_obs_votes,
                            expec_scoring=expec_scoring,
                            verbose=verbose)

    total_time += time.time()

    return {'name': election.name,
            'chain_results': chain_results,
            'time': total_time}


@tf.function
def hmc(n_iter, burn_frac, initial_phc, demo_per_prec, observed_per_prec,
        expec_scoring=False, init_step_size=0.03, adaptation_frac=0.6,
        pause_point=10, verbose=True):
    """
    Run the Hamiltonian Monte Carlo MCMC algorithm to sample the space
    of PHCs in the discrete voter model.

    n_iter (int): the number of iterations to run
    burn_frac (float): the fraction of iterations to burn
    initial_phc (Tensor): the probabilistic hypercube to start with
    observed_per_prec (dict): the number of votes the candidate got in each
    precinct
    demo_per_prec (dict): the precinct-wise demographics of the electorate
    expec_scoring (bool): whether to score by:
        1. the probability of a PHC to produce the outcome
        (False, default)
        2. the difference in the outcome and the PHC's expectation
        (True)
    init_step_size (float): the initial step size for the transition
    adaptation_frac (float): the fraction of the burn in steps to be used
    for step size adaptation
    pause_point (int): the number of iterations to run in each chain chunk
    verbose (bool): whether to display logging and progress bars

    return: a Python dictionary containing the sample, the
    type of scorer, and traces of log probability and log acceptance
    """
    start_time = time.time()
    # Find the number of steps for adaptation
    num_adaptation_steps = int(burn_frac * adaptation_frac * n_iter)

    # Separate the number of iterations into chunks
    fixed_size_steps = n_iter - num_adaptation_steps
    num_chunks = fixed_size_steps // pause_point
    remainder = fixed_size_steps % pause_point

    if verbose:
        print(
            f"This Hamiltonian Monte Carlo chain will be run in {num_chunks} chunks of size {pause_point}, with {num_adaptation_steps} steps of adaptation and {remainder} steps at the end.\n")

    sample_chunks = []
    log_prob_trace_chunks = []
    log_accept_trace_chunks = []

    current_state = initial_phc

    cur_alg_step = 1

    if expec_scoring:
        alg_steps = 3
        scorer = 'expec'

        # Apply `expec_votes` to every precinct
        def expec_log_prob_fn(phc):
            expec_list = []
            for prec, prec_votes in observed_per_prec.items():
                expec_list.append(ev.prob_from_expec(
                    phc, demo_per_prec[prec], prec_votes))

            return tf.math.reduce_mean(expec_list)

        target_log_prob_fn = expec_log_prob_fn

    else:
        alg_steps = 4
        scorer = 'prob'

        if verbose:
            print(
                f"[{cur_alg_step}/{alg_steps}] Creating the binomial coefficients...")
        cur_alg_step += 1

        # Apply `prob_votes` to every precinct
        def prob_log_prob_fn(phc):
            prob_list = []
            for prec, prec_votes in observed_per_prec.items():
                prob_list.append(pv.prob_votes(
                    phc, demo_per_prec[prec], prec_votes))

            return tf.math.reduce_mean(prob_list)

        target_log_prob_fn = expec_log_prob_fn

    # Initialize the adaptive HMC transition kernel
    adaptive_hmc_kernel = init_hmc_kernel(
        target_log_prob_fn, init_step_size, num_adaptation_steps)

    if verbose:
        print(f"[{cur_alg_step}/{alg_steps}] Running the chain for {num_adaptation_steps} steps to adapt the step size...")
    cur_alg_step += 1

    # Run the chain with adaptive HMC to adapt the step size
    if num_adaptation_steps:
        samples, trace = sample_chain(
            adaptive_hmc_kernel,
            num_adaptation_steps,
            current_state,
            trace_fn=hmc_trace_fn)

        step_size_trace = trace[2]

        sample_chunks.append(samples)
        log_prob_trace_chunks.append(trace[0])
        log_accept_trace_chunks.append(trace[1])

        adapted_step_size = tools.find_last_finite(
            step_size_trace, default=init_step_size)
    else:
        adapted_step_size = init_step_size

    # Intialize the HMC transition kernel with the final step size
    hmc_kernel = init_hmc_kernel(target_log_prob_fn, adapted_step_size)

    if verbose:
        print(f"[{cur_alg_step}/{alg_steps}] Running the chain with a step size of {adapted_step_size} on {num_chunks} chunks of {pause_point} iterations each...")
    cur_alg_step += 1

    # Run the chain in chunks to be able to monitor progress
    for i in trange(num_chunks, leave=verbose):
        samples, (log_prob_trace, log_accept_trace, _) = sample_chain(
            adaptive_hmc_kernel,
            pause_point,
            current_state,
            trace_fn=hmc_trace_fn)

        current_state = tf.nest.map_structure(lambda x: x[-1], samples)

        sample_chunks.append(samples)
        log_prob_trace_chunks.append(log_prob_trace)
        log_accept_trace_chunks.append(log_accept_trace)

    if verbose:
        print(
            f"[{cur_alg_step}/{alg_steps}] Running the chain for {remainder} more steps...")
    cur_alg_step += 1

    # Run the chain for the remainder of steps
    samples, (log_prob_trace, log_accept_trace, _) = sample_chain(
        hmc_kernel,
        remainder,
        current_state,
        trace_fn=hmc_trace_fn)

    sample_chunks.append(samples)
    log_prob_trace_chunks.append(log_prob_trace)
    log_accept_trace_chunks.append(log_accept_trace)

    # Consolidate the results
    full_chain = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *sample_chunks)
    full_log_prob_trace = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *log_prob_trace_chunks)
    full_log_accept_trace = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *log_accept_trace_chunks)

    if verbose:
        print(f"[{cur_alg_step}/{alg_steps}] Burning {burn_frac} of samples...")

    burned_chain = burn_in(full_chain, burn_frac)
    burned_log_prob_trace = burn_in(full_log_prob_trace, burn_frac)
    burned_log_accept_trace = burn_in(full_log_accept_trace, burn_frac)

    elapsed = int(time.time() - start_time)
    num_samples = n_iter - int(burn_frac * n_iter)

    if verbose:
        print("Done.")
        print(
            f"Generated a sample of {num_samples} observations in ~{elapsed} seconds.")
    return {'sample': burned_chain,
            'scorer': scorer,
            'log_prob_trace': burned_log_prob_trace,
            'log_accept_trace': burned_log_accept_trace}


@tf.function
def rwm(n_iter, burn_frac, initial_phc, demo_per_prec, observed_per_prec,
        expec_scoring=False, pause_point=10, verbose=True):
    """
    Run the Random Walk Metropolis MCMC algorithm to sample the space
    of PHCs in the discrete voter model.

    n_iter (int): the number of iterations to run
    burn_frac (float): the fraction of iterations to burn
    initial_phc (Tensor): the probabilistic hypercube to start with
    observed_per_prec (dict): the number of votes the candidate got in each
    precinct
    demo_per_prec (dict): the demographics of the electorate, per precinct
    expec_scoring (bool): whether to score by:
        1. the probability of a PHC to produce the outcome
        (False, default)
        2. the difference in the outcome and the PHC's expectation
        (True)
    pause_point (int): the number of iterations to run in each chain chunk
    verbose (bool): whether to display loogging and progress bars

    return: a Python dictionary containing the sample, the
    type of scorer, and traces of log probability and log acceptance
    """
    start_time = time.time()

    # Separate the number of iterations into chunks
    num_chunks = n_iter // pause_point
    remainder = n_iter % pause_point

    if verbose:
        print(
            f"The Random Walk Metropolis chain will be run in {num_chunks} chunks of size {pause_point}, with {remainder} steps at the end.\n")

    sample_chunks = []
    log_prob_trace_chunks = []
    log_accept_trace_chunks = []

    current_state = initial_phc

    cur_alg_step = 1

    if expec_scoring:
        alg_steps = 3
        scorer = 'expec'

        # Apply `prob_from_expec` to every precinct
        def expec_log_prob_fn(phc):
            expec_list = []
            for prec, prec_votes in observed_per_prec.items():
                expec_list.append(ev.prob_from_expec(
                    phc, demo_per_prec[prec], prec_votes))

            return tf.math.reduce_mean(expec_list)

        target_log_prob_fn = expec_log_prob_fn

    else:
        alg_steps = 4
        scorer = 'prob'

        if verbose:
            print(
                f"[{cur_alg_step}/{alg_steps}] Creating the binomial coefficients...")
        cur_alg_step += 1

        # Apply `prob_votes` to every precinct
        def prob_log_prob_fn(phc):
            prob_list = []
            for prec, prec_votes in observed_per_prec.items():
                prob_list.append(pv.prob_votes(
                    phc, demo_per_prec[prec], prec_votes))

            return tf.math.reduce_mean(prob_list)

        target_log_prob_fn = prob_log_prob_fn

    # Initialize the RWM transition kernel
    rwm_kernel = init_rwm_kernel(target_log_prob_fn)

    if verbose:
        print(f"[{cur_alg_step}/{alg_steps}] Running the chain on {num_chunks} chunks of {pause_point} iterations each...")
    cur_alg_step += 1

    # Run the chain in chunks to be able to monitor progress
    for i in trange(num_chunks, leave=verbose):
        samples, (log_prob_trace, log_accept_trace) = sample_chain(
            rwm_kernel,
            pause_point,
            current_state,
            trace_fn=rwm_trace_fn)

        current_state = tf.nest.map_structure(lambda x: x[-1], samples)

        sample_chunks.append(samples)
        log_prob_trace_chunks.append(log_prob_trace)
        log_accept_trace_chunks.append(log_accept_trace)

    if verbose:
        print(
            f"[{cur_alg_step}/{alg_steps}] Running the chain for {remainder} more steps...")
    cur_alg_step += 1

    # Run the chain for the remainder of steps
    samples, (log_prob_trace, log_accept_trace) = sample_chain(
        rwm_kernel,
        remainder,
        current_state,
        trace_fn=rwm_trace_fn)

    sample_chunks.append(samples)
    log_prob_trace_chunks.append(log_prob_trace)
    log_accept_trace_chunks.append(log_accept_trace)

    # Consolidate the results
    full_chain = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *sample_chunks)
    full_log_prob_trace = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *log_prob_trace_chunks)
    full_log_accept_trace = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *log_accept_trace_chunks)

    if verbose:
        print(f"[{cur_alg_step}/{alg_steps}] Burning {burn_frac} of the sample...")

    burned_chain = burn_in(full_chain, burn_frac)
    burned_log_prob_trace = burn_in(full_log_prob_trace, burn_frac)
    burned_log_accept_trace = burn_in(full_log_accept_trace, burn_frac)

    elapsed = int(time.time() - start_time)
    num_samples = n_iter - int(burn_frac * n_iter)

    if verbose:
        print("Done.")
        print(
            f"Generated a sample of {num_samples} observations in ~{elapsed} seconds.")
    return {'sample': burned_chain,
            'scorer': scorer,
            'log_prob_trace': burned_log_prob_trace,
            'log_accept_trace': burned_log_accept_trace}
