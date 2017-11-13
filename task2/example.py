from __future__ import division

import os
import numpy as np
np.random.seed(0)

# ADAM params
ALPHA = 0.001
B1 = 0.9
B2 = 0.999
EPSILON = 1e-8

# general optimization params
EPOCHS = 1
W_WEIGHT = 1.0

# projected SGD params
LAMBDA = 0.001

# feature transformation params
D = 400
NEW_D = 25000


# random Fourier features params
K_WIDTH = 200
OMEGAS = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D)/K_WIDTH,
                                       size=NEW_D)
PHASES = np.random.uniform(0, 2 * np.pi, size=NEW_D)

def transform_(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    np.random.seed(0)
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    Z = np.sqrt(2.0 / NEW_D) * np.cos(np.dot(X, OMEGAS.T) + PHASES)

    return Z


# higher-dimensional mapping params
INDICES = np.random.choice(np.arange((D * (D + 1)) // 2), size=NEW_D)

def transform(X):
    n = X.shape[0]
    Z = np.ndarray((n, NEW_D))

    for i, x in enumerate(X):
        Z[i, :] = np.outer(x, x).flatten()[INDICES]

    return Z


def project_L2(w):
    return w * min(1, 1 / (np.sqrt(LAMBDA) * np.linalg.norm(w, 2)))


def permute(X, Y):
    perm = np.random.permutation(X.shape[0])
    return X[perm, :], Y[perm]


def mapper(key, value):
    # key: None
    # value: one line of input file
    mapper_id = os.getpid()

    X = np.ndarray((len(value), D))
    Y = np.ndarray(len(value))

    for i, val in enumerate(value):
        parts = val.split()
        Y[i], X[i] = (int(float(parts[0])), map(float, parts[1:]))

    X, Y = permute(X, Y)
    X = transform(X)
    n = X.shape[0]

    w = np.zeros(X.shape[1])
    w_hat = np.zeros(X.shape[1])
    m = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])

    for i in range(EPOCHS):
        for t, x_t in enumerate(X):
            y_t = Y[t]
            t = t+1

            if np.dot(x_t, w) * y_t < 1:
                g_t = -y_t * x_t
                m = B1 * m + (1 - B1) * g_t
                m_hat = m / (1 - (B1 ** t))
                v = B2 * v + (1 - B2) * (g_t ** 2)
                v_hat = v / (1 - (B2 ** t))

                w -= ALPHA * m_hat / (np.sqrt(v_hat) + EPSILON)
        w_hat = W_WEIGHT * w + (1 - W_WEIGHT) * w_hat
        print 'Train accuracy (mapper {}, epoch {}): {}'.format(
                mapper_id, i + 1, np.sum(np.dot(X, w_hat) * Y >= 0) / n)

    yield 1, w_hat


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.average(values, 0)
