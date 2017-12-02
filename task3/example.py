from __future__ import division
import numpy as np

from sklearn.cluster.k_means_ import _k_init
from sklearn.utils.extmath import row_norms

DIM = 250
K = 200
ALPHA = 0.75
CORESET_SIZE = 400

def kmeans_loss(X, centers):
    total_loss = 0
    for x in X:
        total_loss += np.min(np.linalg.norm(centers - x, axis=1) ** 2)

    return total_loss


def get_initial_centers(X, n_clusters, init_type='k-means++'):
    if init_type == 'random':
        return X[np.random.choice(X.shape[0], K, replace=False)]

    assert init_type == 'k-means++'
    # centers = [X[np.random.randint(X.shape[0])]]

    # for i in range(SUMMARY_COUNT - 1):
    #     print('Sampled {} centers'.format(i))

    #     centers_arr = np.array(centers)
    #     distances = np.array(
    #         [np.min(np.linalg.norm(centers_arr - x, axis=1) ** 2) for x in X])
    #     probabilities = distances / np.sum(distances)
    #     c = X[np.random.choice(np.arange(X.shape[0]), p=probabilities)]
    #     # TODO: Remove `new_center' from X.
    #     centers.append(c)

    # return np.array(centers)
    return _k_init(X, n_clusters, x_squared_norms=row_norms(X, squared=True),
                   random_state=np.random.RandomState(42))


def kmeans(X, n_clusters=SUMMARY_COUNT, n_init=1, max_iter=20):
    best_centers = None
    best_loss = None

    for rep in range(n_init):
        print('Running repetition {}'.format(rep))
        # initialize cluster centers
        if init_centers is not None:
            centers = init_centers
        else:
            centers = get_initial_centers(X)

        for iter in range(max_iter):
            print('Running iteration {}'.format(iter))
            # assign data points to clusters
            for x in X:
                c = np.argmin(np.linalg.norm(centers - x, axis=1))
                clusters[c].append(x)

            # recalculate clusters
            centers = np.array([np.mean(cluster, axis=0)
                                for cluster in clusters])

        curr_loss = kmeans_loss(X, centers)
        if best_loss is None or curr_loss < best_loss:
            best_centers = centers
            best_loss = curr_loss

    assert best_centers is not None
    return best_centers


def coreset_construction(X):

    n = X.shape[0]
    centers = get_initial_centers(X)

    point_to_cluster = []
    points_in_cluster = [[] for _ in range(K)]
    min_distances = []

    for i, x in enumerate(X):
        distances = np.linalg.norm(centers - x, axis=1)

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

    return X[np.random.choice(n, CORESET_SIZE, replace=False, p=q_x)]


def mapper(key, value):
    # key: None
    # value: one line of input file

    yield 0, coreset_construction(value)


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    coreset = coreset_construction(np.array(values))

    yield kmeans(coreset, n_init=1, max_iter=20)
