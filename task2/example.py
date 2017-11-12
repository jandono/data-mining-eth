from __future__ import division
import numpy as np
np.random.seed(1)

LAMBDA = 0.001
B1 = 0.9
B2 = 0.999
epsilon = 1e-7
alpha = 0.001
new_d = 10000
d = 400

omegas = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=new_d)
phases = np.random.uniform(0, 2 * np.pi, size=new_d)

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    print X.shape
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    Z = np.sqrt(2.0 / new_d) * np.cos(np.dot(X, omegas.T) + phases)

    return Z


# def transform(X):
#     X = (X - np.mean(X, 0)) / np.std(X, 0)
#     n = X.shape[0]
#     d = X.shape[1]

def project_L2(w):
    return w * min(1, 1 / (np.sqrt(LAMBDA) * np.linalg.norm(w, 2)))


def permute(X, Y):
    perm = np.random.permutation(X.shape[0])
    return X[perm, :], Y[perm]


def mapper(key, value):
    # key: None
    # value: one line of input file
    global d
    X = np.ndarray((len(value), d))
    Y = np.ndarray(len(value))

    for i, val in enumerate(value):
        parts = val.split()
        Y[i], X[i] = (int(float(parts[0])), map(float, parts[1:]))

    X = transform(X)
    n = X.shape[0]
    d = X.shape[1]
    w = np.zeros(d)

    X, Y = permute(X, Y)

    m = np.zeros(d)
    v = np.zeros(d)

    for t, x_t in enumerate(X):
        y_t = Y[t]
        t = t+1

        if np.dot(x_t, w) * y_t < 1:
            g_t = -y_t * x_t
            m = B1 * m + (1 - B1) * g_t
            m_hat = m / (1 - (B1 ** t))
            v = B2 * v + (1 - B2) * (g_t ** 2)
            v_hat = v / (1 - (B2 ** t))

            w -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
            w = project_L2(w)

    print np.sum((np.dot(X, w) * Y) >= 0) / n

    yield 1, w


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.average(values, 0)