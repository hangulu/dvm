"""
This module evaluates the Discrete Voter Model for ecological inference and King's Ecological Inference.
"""

import make_grid as mg
import kings_ei as kei
import pymc3 as pm
import elect
import dvm
import tools
import numpy as np
import time
import logging
from tqdm.autonotebook import trange

def dvm_evaluator(election, label, phc_granularity=10, hmc=False,
                  expec_scoring=False, burn_frac=0.3, n_steps=200, n_iter=1):
    """
    Evaluate the accuracy and speed of the Discrete Voter
    Model.

    election (Election): the election to evaluate on
    label (string): the label of the experiment
    phc_granularity (int): the size of a dimension of the PHC
    hmc (bool): whether to use the HMC or RWM kernel
    expec_scoring (bool): whether to score by:
        1. the probability of a PHC to produce the outcome
        (False, default)
        2. the difference in the outcome and the PHC's expectation
        (True)
    burn_frac (float): the fraction of MCMC iterations to burn
    n_steps (int): the number of steps to run the MCMC for
    n_iter (int): the number of times to repeat the experiment

    return: a dictionary of the label, times and MSEs for
    the Discrete Voter Model
    """
    total_time = 0

    total_mle_phc_mse = 0
    total_mean_phc_mse = 0

    initial_grid = mg.make_grid(len(election.demo), phc_granularity)

    for _ in trange(n_iter, desc="Experiment progress"):
        # Get the observed votes for the first candidate
        first_cand = election.candidates[0]
        first_cand_obs_votes = election.outcome[first_cand][0]

        # Run the MCMC with the specified kernel
        total_time -= time.time()

        if hmc:
            chain_results = dvm.hmc(n_steps, burn_frac, initial_grid,
                                    election.demo, first_cand_obs_votes,
                                    expec_scoring=expec_scoring)
        else:
            chain_results = dvm.rwm(n_steps, burn_frac, initial_grid,
                                    election.demo, first_cand_obs_votes,
                                    expec_scoring=expec_scoring)

        total_time += time.time()

        # Find the best grid
        mle_phc = dvm.chain_mle(chain_results)[0]
        mean_phc = dvm.mean_phc(chain_results)

        # Find the most probable cell in the PHC
        best_cell_mle_phc = tools.get_most_probable_cell(mle_phc)
        best_cell_mean_phc = tools.get_most_probable_cell(mean_phc)

        vote_pcts_mle_phc = elect.get_vote_pcts(best_cell_mle_phc, phc_granularity, election.demo)
        vote_pcts_mean_phc = elect.get_vote_pcts(best_cell_mean_phc, phc_granularity, election.demo)

        # Find the MSE of the vote percentages if applicable
        if election.mock:
            # Get the demographic voting probabilities for the first candidate
            dvp_pcts = np.fromiter([pcts[first_cand] for group, pcts in election.dvp.items()], dtype=float)

            mle_phc_mse_array = np.fromiter(vote_pcts_mle_phc.values(), dtype=float)
            mean_phc_mse_array = np.fromiter(vote_pcts_mean_phc.values(), dtype=float)

            total_mle_phc_mse += tools.mse(mle_phc_mse_array, dvp_pcts)
            total_mean_phc_mse += tools.mse(mean_phc_mse_array, dvp_pcts)

    return {'label': label,
            'time': total_time / n_iter,
            'mle_phc_mse': total_mle_phc_mse / n_iter,
            'mean_phc_mse': total_mean_phc_mse / n_iter}


# Suppress logging for pyMC3
pymc3_logger = logging.getLogger('pymc3')
pymc3_logger.setLevel(logging.CRITICAL)

def kei_evaluator(election, label, n_steps=500, n_iter=1):
    """
    Evaluate the accuracy and speed of King's Ecological Inference
    method.

    election (Election): the election to evaluate on
    label (string): the label of the experiment
    n_steps (int): the number of steps to run the MCMC for
    n_iter (int): the number of times to repeat the experiment

    return: a dictionary of the label, times and MSEs for
    the Discrete Voter Model
    """
    # Check if King's EI can be used
    if len(election.demo) > 2:
        raise ValueError("King's Ecological Inference method only works in the 2x2 case.")

    total_time = 0
    total_mse = 0

    for _ in trange(n_iter, desc="Experiment progress"):
        # Get the observed votes for the first candidate
        first_cand = election.candidates[0]
        first_cand_obs_votes = election.outcome[first_cand][0]

        prec_demos = [election.demo]

        # Run King's EI and time it
        total_time -= time.time()

        king_model = kei.eco_inf(prec_demos, first_cand_obs_votes)
        with king_model:
            king_trace = pm.sample(draws=n_steps, progressbar=False)

        total_time += time.time()

        # Find the MSE of the vote percentages if applicable
        if election.mock:
            # Get the demographic voting probabilities for the first candidate
            dvp_pcts = np.fromiter([pcts[first_cand] for group, pcts in election.dvp.items()], dtype=float)

            king_mse_array = np.fromiter([king_trace.get_values('b_1').mean(),
                                      king_trace.get_values('b_2').mean()],
                                     dtype=float)

            total_mse += tools.mse(king_mse_array, dvp_pcts)

    return {'label': label,
            'time': total_time / n_iter,
            'mse': total_mse / n_iter}