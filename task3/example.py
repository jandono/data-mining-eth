import numpy as np
import os

DIM = 250
SUMMARY_COUNT = 200


def kmeans_loss(X, centers):
    total_loss = 0
    for x in X:
        total_loss += np.min(np.linalg.norm(centers - x, axis=1) ** 2)

    return total_loss


def get_initial_centers(X, init_type='k-means++'):
    if init_type == 'random':
        return X[np.random.choice(X.shape[0], SUMMARY_COUNT, replace=False)]

    assert init_type == 'k-means++'
    centers = [X[42]]

    for i in range(SUMMARY_COUNT - 1):
        # print('Sampled {} centers'.format(i))

        centers_arr = np.array(centers)
        distances = np.array(
            [np.min(np.linalg.norm(centers_arr - x, axis=1) ** 2) for x in X])
        probabilities = distances / np.sum(distances)

        c = X[np.random.choice(np.arange(X.shape[0]), p=probabilities)]
        # TODO: Remove `new_center' from X.
        centers.append(c)

    return np.array(centers)


def kmeans(X, n_clusters=SUMMARY_COUNT, n_init=10, max_iter=20):
    best_centers = None
    best_loss = None

    for rep in range(n_init):
        # print('Running repetition {}'.format(rep))
        # initialize cluster centers
        centers = get_initial_centers(X)

        clusters = [[] for _ in range(SUMMARY_COUNT)]
        for iter in range(max_iter):
            # print('Running iteration {}'.format(iter))
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


def mapper(key, value):
    # key: None
    # value: one line of input file
    # kmeans = KMeans(init='k-means++', n_clusters=SUMMARY_COUNT,
    #                 n_init=10, max_iter=10)
    # kmeans.fit(value)
    # print('Mapper {} finished'.format(os.getpid()))
    # yield 0, kmeans.cluster_centers_
    yield 0, value


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    # kmeans = KMeans(init='random', n_clusters=SUMMARY_COUNT,
    #                 n_init=1, max_iter=20)
    # kmeans.fit(values)
    # yield kmeans.cluster_centers_
    yield kmeans(np.array(values), n_init=1)
