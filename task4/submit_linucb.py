from __future__ import division

from numpy.linalg import inv
import numpy as np


##########################################
# Constants & hyper-parameters
##########################################

# the desired probability threshold which controls how close to the true
# conditional reward mean we want to be (i.e. E[r_{t,a} | x_{t,a}])
# NOTE: fine-tune it!

DELTA = 0.35


# Alpha constant which controls the exploration/exploitation ratio
#
# In the proofs, this is normally computed from DELTA, the desired
# probability threshold for how close we are to the conditional reward
# expectation, as follows:
#
#   ALPHA = 1.0 + np.sqrt(np.log(2.0 / DELTA) / 2.0)
#
# but in practice it is tuned on its own.
#
# Some results on the _online_ data set with different values of alpha:
#   * 0.1:      0.0601379310345
#   * 0.15:     0.064856235487
#   * 0.195:    0.0676799257879
#   * 0.2:      0.0658229480017
#   * 0.25:     0.0658113668316 (also with reward set to -1 when it is 0)
#   * 0.3:      0.0652978252921
#   * 0.4:      0.0643042207167
#   * 0.5:      0.0638841043307
#   * derived from DELTA = 0.01 (alpha = ~2.6): 0.0523131991051
#
# NOTE: More indepth search for alpha was performed locally, checking values from 0.1 to 0.3 with a step of 0.005

ALPHA = 0.195
# ALPHA = 1.0 + np.sqrt(np.log(2.0 / DELTA) / 2.0)


# dimensionality of the `context' (i.e. user features)
# it also happens to be the dimensionality of the article features, so we are
# using it for both of them throughout the implementation

D = 6


##########################################
# Algorithm-specific data structures
##########################################
# NOTE: paper notations are preferred where possible, in what follows

# the user feature vector in the last trial
# type: np.ndarray

x = None


# the identifier of the articles chosen in the last trial
# type: int

at = None


# the feature vectors of the articles; should not be changed after set by
# `set_articles'.
# type: dict(int, np.ndarray)
#
# NOTE: it is initialized, together with all the following dicts, in the
# `set_articles' function
# Not used for the LinUCB algorithm, only in the Hybrid

Z = {}


# the timestamp to "time" map; we consider the events that take place at the
# same timestamp as having the same time

ts_to_time = {}
last_time = 1


# the matrices/vectors with the same name in the algorithm; there is one for
# each article
# type: dict(int, np.ndarray)

A = {}
Ai = {}
b = {}


# regularization for the linear regression solution
lambda_reg = 1.0


# Initial decaying factor for Alpha
INITIAL_DECAYING_FACTOR = 2


##########################################
# Interface functions
##########################################

def set_articles(articles):
    # declare the global variables changed in this function
    global Z, A, b

    for article, features in articles.items():
        # initialize all the articles-indexed data structures defines above
        Z[article] = np.array(features)
        # steps 5, 6 in Algorithm 1
        A[article] = np.eye(D) * lambda_reg
        Ai[article] = inv(A[article])
        b[article] = np.zeros(D)


def update(reward):
    # declare the global variables changed in this function
    global A, b, x

    # Try to see how changing the rewards influences the performance of the algorithm
    #
    # if reward == 0:
    #     reward = -1
    #
    # if reward > 0:
    #     reward = 20  # this can be tuned

    # ignore non-matching lines
    # From the paper:
    #   Otherwise, if the policy π selects a different arm from the one that
    #   was taken by the logging policy, then the event is entirely ignored,
    #   and the algorithm proceeds to the next event without any other change
    #   in its state.
    if reward < 0:
        return

    # lines 12-13 from Algorithm 1
    A[at] += np.outer(x, x.T)
    Ai[at] = inv(A[at])
    b[at] += reward * x


def recommend(time, user_features, choices):
    # declare the global variables changed in this function
    global ts_to_time, last_time, x, at

    # map this timestamp to a time
    if time not in ts_to_time:
        ts_to_time[time] = last_time
        last_time += 1

    # compute ALPHA for this recommendation round
    # Decaying alpha
    alpha = ALPHA * (1 + INITIAL_DECAYING_FACTOR / ts_to_time[time])

    # Nondecaying alpha
    # alpha = ALPHA

    # line 2 from Algorithm 1: observing the features
    # NOTE: in the algorithm we have such an x for each articles; in our
    # specific case, however, it is the same for all of them: the user features
    x = np.asarray(user_features)

    # lines 3 to 10: iterate through all actions (i.e. articles)
    p_t = {}  # stores the CI upper bound of the prediction, for all choices
    for a in choices:
        # line 8
        theta_hat = np.dot(Ai[a], b[a])
        # line 9
        p_t[a] = np.dot(theta_hat.T, x) + alpha * np.sqrt(x.T.dot(Ai[a]).dot(x))

    # line 11: choose the best one according to the UCB approach
    at = max(p_t, key=p_t.get)

    return at
