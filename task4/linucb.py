from __future__ import division

from numpy.linalg import inv, multi_dot as mult
import numpy as np


##########################################
## Constants & hyper-parameters
##########################################

# the desired probability threshold which controls how close to the true
# conditional reward mean we want to be (i.e. E[r_{t,a} | x_{t,a}])
# NOTE: fine-tune it!
DELTA = 0.05

# constant (based on DELTA) which ensures the confidence bounds are correctly
# computed by the algorithm
# NOTE: it might make more sense to fine-tune directly this one, instead of
# DELTA
# ALPHA = 1.0 + np.sqrt(np.log(2.0 / DELTA) / 2.0)
ALPHA = 0.1

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

# the matrices/vectors with the same name in the algorithm; there is one for
# each article
# type: dict(int, np.ndarray)
A = {}
Ai = {}
b = {}


##########################################
## Interface functions
##########################################

def set_articles(articles):
    # declare the global variables changed in this function
    global Z, A, B, b

    for article, features in articles.items():
        # initialize all the articles-indexed data structures defines above
        Z[article] = np.array(features)
        # steps 8,9,10 in Algorithm 2
        A[article] = np.eye(D)
        Ai[article] = np.eye(D)
        b[article] = np.zeros(D)


def update(reward):
    # declare the global variables changed in this function
    global A, b

    # TODO(ccruceru): or maybe we shouldn't ignore the '-1' update calls, but
    # change them to 0? comment the one from above if this is uncommented!
    # reward = max(reward, 0)

    # ignore non-matching lines
    # TODO(ccruceru): *&*@#?! in the description they say that the line is
    # discarded, why do they call us with a negative reward?
    if reward < 0:
        return

    # NOTE: this is experimental: try to see what happens if we penalize more
    # for mistakes; this only works if the first option from above is used
    # if reward == 0:
    #     reward = -1

    # lines 12-13 from Algorithm 1
    # TODO(ccruceru): cache the repeated matrix multiplications
    A[at] += np.outer(x, x.T)
    Ai[at] = inv(A[at])
    b[at] += reward * x


def recommend(time, user_features, choices):
    # TODO(ccruceru): do we need the time?

    # declare the global variables changed in this function
    global x, at

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
        p_t[a] = np.dot(theta_hat.T, x) + ALPHA * np.sqrt(mult([x.T, Ai[a], x]))

    # line 16: choose the best one according to the UCB approach
    at = max(p_t, key=p_t.get)

    return at
