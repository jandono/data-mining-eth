from __future__ import division

from numpy.linalg import inv, multi_dot as mult
import numpy as np


##########################################
## Constants & hyper-parameters
##########################################

# the desired probability threshold which controls how close to the true
# conditional reward mean we want to be (i.e. E[r_{t,a} | x_{t,a}])
# NOTE: fine-tune it!
DELTA = 0.01

# constant (based on DELTA) which ensures the confidence bounds are correctly
# computed by the algorithm
ALPHA = 1 + np.sqrt(np.log(2 / DELTA) / 2)

# dimensionality of the `context' (i.e. user features)
# it also happens to be the dimensionality of the article features, so we are
# using with for both of them throughout the implementation
D = 6

# dimensionality of the user/article combination (i.e. size of the linearized
# outer product matrix between the two)
K = D * D


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

# the user/chosen articles combination feature vector
# NOTE: this is essentially np.outer(x, Z[at]).flatten(), where `x' is the one
# from above, not an arbitrary user; we store it here nonetheless to avoid
# computing it again in the `update()' function.
# type: np.ndarray
z = None

# the feature vectors of the articles; should not be changed after set by
# `set_articles'. The z_{t,a} from the paper are obtained as follows:
#
#       z_ta = np.outer(Z[i], x.T).flatten()
#
# type: dict(int, np.ndarray)
#
# NOTE: it is initialized, together with all the following dicts, in the
# `set_articles' function
# TODO(ccruceru): index articles from 0 to N and make this a ndarray if somehow
# we need it
Z = {}

# the matrix with the same name in the hybrid LinUCB algorithm and its inverse
# type: np.ndarray
#
# NOTE: we follow the advice from paper and cache the inverses for A0 and Aa
# (see below)
A0 = np.eye(K)
A0i = np.eye(K)

# the vector with the same name in the hybrid LinUCB algorithm
# type: np.ndarray
b0 = np.zeros(K)

# the matrices/vectors with the same name in the algorithm; there is one for
# each article
# type: dict(int, np.ndarray)
A = {}
Ai = {}
B = {}
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
        B[article] = np.zeros((D, K))
        b[article] = np.zeros(D)


def update(reward):
    # declare the global variables changed in this function
    global A0, A0i, b0, A, Ai, B, b

    # lines 17-23 from Algorithm 2
    # TODO(ccruceru): cache the repeated matrix multiplications
    A0 += mult([B[at].T, Ai[at], B[at]])
    b0 += mult([B[at].T, Ai[at], b[at]])
    A[at] += np.outer(x, x.T)
    Ai[at] = inv(A[at])
    B[at] += np.outer(x, z.T)
    b[at] += reward * x
    A0 += np.outer(z, z.T) - mult([B[at].T, Ai[at], B[at]])
    A0i = inv(A0)
    b0 += reward * z - mult([B[at].T, Ai[at], b[at]])


def recommend(time, user_features, choices):
    # TODO(ccruceru): do we need the time?

    # declare the global variables changed in this function
    global x, at, z

    # line 4: observing the features
    # NOTE: in the algorithm we have such an x for each articles; in our
    # specific case, however, it is the same for all of them: the user features
    x = np.asarray(user_features)
    # but we do have different client/article feature vectors
    Z_t = {i: np.outer(Z[i], x.T).flatten() for i in Z}

    # line 5: compute beta hat
    beta_hat = np.dot(A0i, b0)

    # lines 6 to 15: iterate through all actions (i.e. articles)
    p_t = {}  # stores the CI upper bound of the prediction, for all choices
    for a in choices:
        # line 12
        theta_hat = np.dot(Ai[a], b[a] - np.dot(B[a], beta_hat))
        # line 13
        s_ta = mult([Z_t[a].T, A0i, Z_t[a]]) + \
                - 2 * mult([Z_t[a].T, A0i, B[a].T, Ai[a], x]) + \
                + mult([x.T, Ai[a], x]) + \
                + mult([x.T, Ai[a], B[a], A0i, B[a].T, Ai[a], x])
        # line 14
        p_t[a] = np.dot(Z_t[a].T, beta_hat) + \
                + np.dot(x.T, theta_hat) + \
                + ALPHA * np.sqrt(s_ta)

    # line 16: choose the best one according to the UCB approach
    at = max(p_t, key=p_t.get)
    # save the corresponding `z' vector
    z = Z_t[at]

    return at
