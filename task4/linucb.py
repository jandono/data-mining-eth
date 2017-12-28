from __future__ import division

from numpy.linalg import inv
import numpy as np


##########################################
## Constants & hyper-parameters
##########################################

# the desired probability threshold which controls how close to the true
# conditional reward mean we want to be (i.e. E[r_{t,a} | x_{t,a}])
# NOTE: fine-tune it!
DELTA = 0.01

# constant which controls the exploration/exploitation ratio
#
# in the proofs, this should normally be computed from DELTA, the desired
# probability threshold for how close we are to the conditional reward
# expectation, as follows:
#
#   ALPHA = 1.0 + np.sqrt(np.log(2.0 / DELTA) / 2.0)
#
# but in practice it is tuned on its own.
#
# Some results on the _online_ data set:
#   * 0.1:  0.0601379310345
#   * 0.15: 0.064856235487
#   * 0.2:  0.0658229480017
#   * 0.25: 0.0658113668316 (also with reward set to -1 when it is 0)
#   * 0.3:  0.0652978252921
#   * 0.4:  0.0643042207167
#   * 0.5:  0.0638841043307
#   * derived from DELTA = 0.01 (it's ~2.6): 0.0523131991051
#
# These value give completely different results on the local data set; for
# instnace, 0.1 is the best, with CTR = 0.0463.
# ALPHA = 0.1
#
# Some results on the large local data set for different alpha
#
#   * 0.1:      0.06003
#   * 0.15:     0.06003
#   * 0.2:      0.06807
#   * 0.3:      0.06696
#
#
#
#
# ALPHA = 0.15
ALPHA = None
# ALPHA = 1.0 + np.sqrt(np.log(2.0 / DELTA) / 2.0)

# dimensionality of the `context' (i.e. user features)
# it also happens to be the dimensionality of the article features, so we are
# using with for both of them throughout the implementation
D = 6


##########################################
## Algorithm-specific data structures
##########################################
## NOTE: paper notations are preferred where possible, in what follows

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


##########################################
## Interface functions
##########################################

def set_articles(articles, a):
    # declare the global variables changed in this function
    global Z, A, B, b, ALPHA

    ALPHA = a

    for article, features in articles.items():
        # initialize all the articles-indexed data structures defines above
        Z[article] = np.array(features)
        # steps 8,9,10 in Algorithm 2
        A[article] = np.eye(D)
        Ai[article] = np.eye(D)
        b[article] = np.zeros(D)


def update(reward):
    # declare the global variables changed in this function
    global A, b, x

    # ignore non-matching lines
    # TODO(ccruceru): *&*@#?! in the description they say that the line is
    # discarded, why do they call us with a negative reward? From the paper:
    #
    #   Otherwise, if the policy π selects a different arm from the one that
    #   was taken by the logging policy, then the event is entirely ignored,
    #   and the algorithm proceeds to the next event without any other change
    #   in its state.
    if reward < 0:
        return

    # NOTE: this is experimental: try to see what happens if we penalize more
    # for mistakes; this only works if the first option from above is used
    # if reward == 0:
    #     reward = -1

    # lines 12-13 from Algorithm 1
    A[at] += np.outer(x, x.T)
    Ai[at] = inv(A[at])
    b[at] += reward * 0.01 * last_time * x


def recommend(time, user_features, choices):
    # TODO(ccruceru): do we need the time?

    # declare the global variables changed in this function
    global ts_to_time, last_time, x, at

    # map this timestamp to a time
    if time not in ts_to_time:
        ts_to_time[time] = last_time
        last_time += 1

    # compute ALPHA for this recommendation round
    # TODO(ccruceru): this is an attempt to make it decay in time
    alpha = ALPHA * (1 + 2 / ts_to_time[time])
    # alpha = ALPHA

    # line 2: observing the features
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
