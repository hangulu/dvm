"""
This module implements the prob_votes subroutine for the Discrete Voter Model
for ecological inference.
"""

import functools
import tensorflow as tf
import tensorflow_probability as tfp

import elect
import tools


@functools.lru_cache(maxsize=None)
def get_vote_probability(flat_index, phc_shape, demo, partitions):
    """
    Find the probability of a PHC's cell producing a
    vote outcome of a given election for a candidate,
    with a given PHC.

    flat_index (int): the flat index of the selected cell
    phc_shape (tuple): the shape of a PHC's Tensor representation
    demo (tuple): the demographics of the district
    partitions (tuple): the partitions of votes for a candidate

    return: the probability that a PHC's cell produced the observed outcome
    """
    # Find the corresponding index
    index = tf.unravel_index(flat_index, phc_shape)
    matrix_dim = phc_shape[0]

    # Find the vote percentages for each demographic group
    vote_pcts = elect.get_vote_pcts_list(index, matrix_dim)

    # Binomial calculation
    # Independent binomial distributions for each demographic group where each
    # represents the probability of the voters in that group voting together
    # to satisfy the possible partitions of voters
    pmf = tfp.distributions.Binomial(demo, probs=vote_pcts).prob(partitions)

    return tf.math.reduce_sum(tf.math.reduce_prod(pmf, 1))


def prob_votes(phc, demo, observed, rwm=False):
    """
    Find the probability that a PHC produced
    the observed number of votes that a candidate
    received in a given election, with a given
    PHC.

    phc (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district
    observed (int): the observed number of votes the candidate received
    rwm (bool): whether this function serves the RWM or HMC kernel

    return: the probability that a PHC produced the observed outcomes
    """
    partitions = tuple(tools.permute_integer_partition(observed, len(demo)))

    if rwm:
        flat_phc = tf.reshape(phc, [-1])

        get_vote_prob_partial = functools.partial(
            get_vote_probability,
            phc_shape=tuple(phc.shape),
            demo=tuple(demo.values()),
            partitions=partitions)
    else:
        normalized_phc = tools.prob_normalize(phc)
        flat_phc = tf.reshape(normalized_phc, [-1])

        get_vote_prob_partial = functools.partial(
            get_vote_probability,
            phc_shape=tuple(normalized_phc.shape),
            demo=tuple(demo.values()),
            partitions=partitions)

    # Calculate the probability for each cell
    vote_prob = [get_vote_prob_partial(flat_index) for flat_index in range(tf.size(flat_phc))]

    # TODO: Find a way to vectorize the above operation using Tensors
    # vote_prob = tf.map_fn(get_vote_prob_partial,
    #                       tf.range(tf.size(flat_phc)), dtype=tf.float32)

    # tf.math.multiply(vote_prob, flat_phc): the vector of probabilities that
    # each of the events happened (where an event is a cell producing the
    # vote outcome).
    # 1 - tf.math.multiply(vote_prob, flat_phc): the vector of probabilities
    # that each of the events did not happen
    # tf.math.reduce_prod(1 - tf.math.multiply(vote_prob, flat_phc)):
    # the probability that none of the events happened
    phc_prob_complement = tf.math.reduce_prod(
        1 - tf.math.multiply(vote_prob, flat_phc))

    # 1 - phc_prob_complement: the probability that at least one of the events
    # happened
    return tf.math.log(1 - phc_prob_complement)
