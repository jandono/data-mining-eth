from __future__ import division
import numpy as np

DIM = 250
K = 200
ALPHA = 16 * (np.log(K) + 2)
CORESET_SIZE = 2000

def kmeans_loss(X, centers):
    total_loss = 0
    for x in X:
        total_loss += np.min(np.linalg.norm(centers - x, ord=2, axis=1) ** 2)

    return total_loss


def get_initial_centers(X, init_type='k-means++'):
    if init_type == 'random':
        return X[np.random.choice(X.shape[0], K, replace=False)]

    assert init_type == 'k-means++'

    centers = [X[np.random.randint(X.shape[0])]]
    distances = None
    for i in range(K - 1):
        #print('Sampled {} centers'.format(i))
        distance_new_center = np.linalg.norm(-X + np.array(centers[i]), ord=2, axis=1) ** 2
        #distance_new_center = [(np.linalg.norm(np.array(centers[i]) - x, ord=2) ** 2) for x in X]

        if distances is None:
            distances = distance_new_center
        else:
            distances = np.minimum(distances, distance_new_center)

        sum_distances = np.sum(distances)
        probabilities = distances / sum_distances
        cluster_index = np.random.choice(X.shape[0], p=probabilities)
        c = X[cluster_index]
        X = np.delete(X, cluster_index, axis=0)
        distances = np.delete(distances, cluster_index, axis=0)
        centers.append(c)

    return np.array(centers)


def kmeans(X, n_init=1, max_iter=20, init_centers=None):
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

        clusters = [[] for _ in range(K)]
        prev_centers = None
        for iter in range(max_iter):
            # print('Running iteration {}'.format(iter))

            # # assign data points to clusters
            # z_kn = np.zeros((K, N))
            #
            # for i, x in enumerate(X):
            #     c = np.argmin(np.linalg.norm(centers - x, axis=1))
            #     z_kn[c, i] = 1
            #
            # for k in range(K):
            #     centers[k] = np.dot(z_kn[k, :], X) / np.sum(z_kn[k, :])

            # assign data points to clusters
            for x in X:
                c = np.argmin(np.linalg.norm(centers - x, ord=2, axis=1))
                clusters[c].append(x)

            # recalculate clusters
            centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])

            if prev_centers is not None and np.array_equal(centers, prev_centers):
                break

            prev_centers = centers

        curr_loss = kmeans_loss(X, centers)
        if best_loss is None or curr_loss < best_loss:
            best_centers = centers
            best_loss = curr_loss

    return best_centers


def coreset_construction(X, size, replace=False):
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

    return X[np.random.choice(n, size, replace=replace, p=q_x)]


def mapper(key, value):
    # key: None
    # value: one line of input file

    #yield 0, coreset_construction(np.array(value), CORESET_SIZE)
    yield 0, np.array(value)


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    #coreset = coreset_construction(np.array(values), CORESET_SIZE)

    yield kmeans(values, n_init=1, max_iter=20)
