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
W_WEIGHT = 1

# projected SGD params
LAMBDA = 0.001

# feature transformation params
D = 400
NEW_D = 37000


###############################################################################
# APPROACH 1: Random Fourier Features

# params
K_WIDTH = 200  # controls the RBF width (usually `gamma')
OMEGAS = np.random.multivariate_normal(mean=np.zeros(D),
                                       cov=np.eye(D)/K_WIDTH,
                                       size=NEW_D)
PHASES = np.random.uniform(0, 2 * np.pi, size=NEW_D)

# transformation function
def transform_(X):
    """Generates features constructed from samples drawn from the frequency
    representation of the RBF kernel.
    """
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    Z = np.sqrt(2.0 / NEW_D) * np.cos(np.dot(X, OMEGAS.T) + PHASES)

    return Z

###############################################################################


###############################################################################
# APPROACH 2: Second-order polynomial features

def transform(X):
    """Transforms to the first `NEW_D' second-order polynomial components.
    """
    # make sure this function works with both 1D (including Python lists) and
    # 2D arrays
    if type(X) == list:
        X = np.array(X)
    if X.ndim == 1:
        return np.outer(X, X).flatten()[:NEW_D]

    n = X.shape[0]
    Z = np.ndarray((n, NEW_D))
    for i, x in enumerate(X):
        Z[i, :] = np.outer(x, x).flatten()[:NEW_D]

    return Z

###############################################################################


def project_L2(w):
    """Projects the given w vector on the feasible set determined by the
    hyper-parameter LAMBDA.

    NOTE: This function is not used in the final version as the validation gave
    better results when the parameters of the models were not projected back to
    the feasible set.
    """
    return w * min(1, 1 / (np.sqrt(LAMBDA) * np.linalg.norm(w, 2)))


def permute(X, Y):
    """Randomly permutes the given data set (both the explanatory and the
    response variables).
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm, :], Y[perm]


def mapper(key, value):
    mapper_id = os.getpid()

    # get the Xs and Ys from the given subset
    X = np.ndarray((len(value), D))
    Y = np.ndarray(len(value))

    for i, val in enumerate(value):
        parts = val.split()
        Y[i], X[i] = (int(float(parts[0])), map(float, parts[1:]))

    # do the feature transformation
    X = transform(X)
    n, d = X.shape

    # prepare optimization (ADAM) specific variables
    w = np.zeros(d)     # the SVM params
    w_hat = np.zeros(d) # the across-epochs averaged SVM params
    m = np.zeros(d)     # Nesterov's momentum
    v = np.zeros(d)     # AdaGrad's learning rate decay factor

    for i in range(EPOCHS):
        # shuffle after every epoch
        X, Y = permute(X, Y)

        # go through one point at a time (SGD)
        for t, (x_t, y_t) in enumerate(zip(X, Y)):
            t = t + 1

            if np.dot(x_t, w) * y_t < 1:
                g_t = -y_t * x_t
                m = B1 * m + (1 - B1) * g_t
                m_hat = m / (1 - (B1 ** t))
                v = B2 * v + (1 - B2) * (g_t ** 2)
                v_hat = v / (1 - (B2 ** t))

                w -= ALPHA * m_hat / (np.sqrt(v_hat) + EPSILON)
            # if it is not misclassified, the gradient is 0 and there is
            # nothing else to do

        # update the across-epochs SVM params
        w_hat = W_WEIGHT * w + (1 - W_WEIGHT) * w_hat

        # trace output
        print 'Train accuracy (mapper {}, epoch {}): {}'.format(
                mapper_id, i + 1, np.sum(np.dot(X, w_hat) * Y >= 0) / n)

    # our final parameters are given by the across-epochs aggregated params
    yield 1, w_hat


def reducer(key, values):
    # the reducer simply computes the average of the SVM params computed by
    # mappers
    yield np.average(values, 0)


# here we just make sure that our transform functions work for both 1D and 2D
# arrays
if __name__ == '__main__':
    print transform(np.arange(D)).shape
    print transform(np.arange(D).tolist()).shape
    print transform_(np.arange(D)).shape
    print transform_(np.arange(D).tolist()).shape
