from __future__ import division
import numpy as np
from scipy.spatial.distance import cdist

DIM = 250
K = 200
ALPHA = 16 * (np.log(K) + 2)
CORESET_SIZE = 2000


def kmeans_loss(X, centers, distances=None):
    """
    Computes the kmeans loss, given data X and centers
    Optionally it can get precomputed distances
    """

    if distances is None:
        distances = cdist(X, centers, 'sqeuclidean')

    loss = np.sum(np.min(distances, axis=1))
    return loss


def get_initial_centers(X, init_type='k-means++'):
    """
    Given data X calculates the initial centers
    Two options available:
        1. Sample centers uniformly at random
        2. Sample centers using k-means++ initialisation
    """

    if init_type == 'random':
        return X[np.random.choice(X.shape[0], K, replace=False)]

    assert init_type == 'k-means++'

    centers = [X[np.random.randint(X.shape[0])]]

    distances = None
    for i in range(K - 1):
        # print('Sampled {} centers'.format(i))
        differences = centers[i] - X
        distance_new_center = np.linalg.norm(differences, ord=2, axis=1) ** 2

        if distances is None:
            distances = distance_new_center
        else:
            distances = np.minimum(distances, distance_new_center)

        probabilities = distances / np.sum(distances)
        cluster_index = np.random.choice(X.shape[0], p=probabilities)
        c = X[cluster_index]
        centers.append(c)

    return np.array(centers)


def kmeans(X, n_init=1, max_iter=20, init_centers=None):
    """
    Applies the k-means algorithm to cluster data X into K centers
    """

    best_centers = None
    best_loss = None

    N = X.shape[0]

    for rep in range(n_init):
        # print('Running repetition {}'.format(rep))
        # initialize cluster centers
        if init_centers is not None:
            centers = init_centers
        else:
            centers = get_initial_centers(X)

        prev_labels = None
        indices = np.arange(N)
        for iter in range(max_iter):
            print('Running iteration {}'.format(iter))

            # assign data points to clusters
            z_kn = np.zeros((K, N), dtype=np.bool)
            distances = cdist(X, centers, 'sqeuclidean')
            labels = distances.argmin(axis=1)

            if np.array_equal(labels, prev_labels):
                print('early stopping at iteration {}'.format(iter))
                break

            prev_labels = labels
            z_kn[labels, indices] = 1

            # recalculate centers
            centers = np.divide(np.dot(z_kn, X).T, np.sum(z_kn, axis=1)).T
            # for c in range(K):
            #     centers[c] = np.mean(X[labels == c], axis=0)

        curr_loss = kmeans_loss(X, centers, distances=distances)
        if best_loss is None or curr_loss < best_loss:
            best_centers = centers
            best_loss = curr_loss

    return best_centers


def kmedians(X, n_init=1, max_iter=20, init_centers=None):
    """
    Applies the k-medians algorithm to cluster data X into K centers
    """

    best_centers = None
    best_loss = None

    N = X.shape[0]

    for rep in range(n_init):
        # print('Running repetition {}'.format(rep))
        # initialize cluster centers
        if init_centers is not None:
            centers = init_centers
        else:
            centers = get_initial_centers(X)

        prev_labels = None
        for iter in range(max_iter):
            # print('Running iteration {}'.format(iter))

            # assign data points to clusters
            distances = cdist(X, centers, 'cityblock')
            labels = distances.argmin(axis=1)

            if np.array_equal(labels, prev_labels):
                print('early stopping at iteration {}'.format(iter))
                break

            prev_labels = labels

            # recalculate clusters
            for c in range(K):
                centers[c] = np.median(X[labels == c], axis=0)

        curr_loss = kmeans_loss(X, centers)
        if best_loss is None or curr_loss < best_loss:
            best_centers = centers
            best_loss = curr_loss

    return best_centers


def coreset_construction(X, coreset_size, replace=False):
    """
    Given data X returns a coreset of size coreset_size
    """

    n = X.shape[0]
    centers = get_initial_centers(X)

    point_to_cluster = []
    points_in_cluster = [[] for _ in range(K)]
    min_distances = []

    for i, x in enumerate(X):
        distances = np.linalg.norm(centers - x, ord=2, axis=1)

        # For each x compute distances to the closest center
        min_distances.append(np.min(distances))

        # Compute Bx for each x
        c = np.argmin(distances)
        point_to_cluster.append(c)
        points_in_cluster[c].append(i)

    sum_bx = []
    for i, x in enumerate(X):
        # Compute sum(Bx) for each x
        cluster_x = point_to_cluster[i]
        indices = points_in_cluster[cluster_x]
        sum_bx.append(np.sum([min_distances[index] for index in indices]))

    # Compute CF
    c_phi = np.sum(min_distances) / n

    # Compute qx for each x
    q_x = []

    for i, x in enumerate(X):
        term1 = ALPHA * min_distances[i] / c_phi
        bx_cardinality = len(points_in_cluster[point_to_cluster[i]])
        term2 = (2 * ALPHA * sum_bx[i]) / (bx_cardinality * c_phi)
        term3 = 4 * n / bx_cardinality

        q_x.append(term1 + term2 + term3)

    q_x /= np.sum(q_x)

    return X[np.random.choice(n, coreset_size, replace=replace, p=q_x)]


def mapper(key, value):
    """
    The mapper can either yield a coreset of the data provided or yield the entire data
    """

    yield 0, coreset_construction(value, CORESET_SIZE)
    # yield 0, value


def reducer(key, values):
    """
    Coreset construction can be used here as well to modify summarize the data provided
    The reducer then runs either kmeans or kmedians on the data
    """

    # coreset = coreset_construction(np.array(values), CORESET_SIZE)

    yield kmeans(values, n_init=2, max_iter=100)
    # yield kmedians(values, n_init=2, max_iter=100)
