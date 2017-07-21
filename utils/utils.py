import os
import numpy as np


def mkdir_or_not(d: str):
    if not os.path.exists(d):
        os.mkdir(d)


def weighted_pick(probabilities, vocabulary_size, topn=1):
    """Roll the dice to produce a random integer in the [0..ALPHASIZE] range,
    according to the provided probabilities. If topn is specified, only the
    topn highest probabilities are taken into account.
    :param vocabulary_size:
    :param probabilities: a list of size ALPHASIZE with individual probabilities
    :param topn: the number of highest probabilities to consider. Defaults to all of them.
    :return: a random integer
    """
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(vocabulary_size, 1, p=p)[0]


def weighted_pick_v2(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return int(np.searchsorted(t, np.random.rand(1) * s))