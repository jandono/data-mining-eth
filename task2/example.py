from __future__ import division
import numpy as np
# from sklearn.svm import SVC

LAMBDA = 0.001
B1 = 0.9
B2 = 0.999
EPSILON = 1e-7
ALPHA = 0.05
NEW_D = 15000
D = 400



def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    np.random.seed(0)
    omegas = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D)/150, size=NEW_D)
    # omegas = np.random.standard_cauchy((NEW_D, D))
    phases = np.random.uniform(0, 2 * np.pi, size=NEW_D)

    X = (X - np.mean(X, 0)) / np.std(X, 0)
    Z = np.sqrt(2.0 / NEW_D) * np.cos(np.dot(X, omegas.T) + phases)

    return Z


def rbf_kernel(x, y, sigma_s=1):

    return np.exp(-(np.linalg.norm(x - y, 2) ** 2) / 2*sigma_s)


def project_L2(w):
    return w * min(1, 1 / (np.sqrt(LAMBDA) * np.linalg.norm(w, 2)))


def permute(X, Y):
    perm = np.random.permutation(X.shape[0])
    return X[perm, :], Y[perm]


def mapper(key, value):
    # key: None
    # value: one line of input file
    X = np.ndarray((len(value), D))
    Y = np.ndarray(len(value))

    for i, val in enumerate(value):
        parts = val.split()
        Y[i], X[i] = (int(float(parts[0])), map(float, parts[1:]))

    X, Y = permute(X, Y)

    X = transform(X)
    n = X.shape[0]

    # svm = SVC()
    # svm.fit(X, Y)
    #
    # print svm.score(X, Y)

    w = np.zeros(X.shape[1])
    w_hat = np.zeros(X.shape[1])

    m = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])

    # for i in range(5):
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
            # w = project_L2(w)

            # w_hat = 0.8 * w_hat + 0.2 * w
            # w_hat = ((t-1) * w_hat + w) / t
    print np.sum((np.dot(X, w) * Y) >= 0) / n


    yield 1, w


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    yield np.average(values, 0)