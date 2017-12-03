import numpy as np

X = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5]])

K = 3

centers = np.array([X[np.random.randint(X.shape[0])]])
distances = np.array([])

choice_array = np.arange(X.shape[0])
sum_distances = 0
for i in range(K - 1):
    distance_new_center = np.min([(np.linalg.norm(centers[i] - x) ** 2) for x in X])
    np.append(distances, distance_new_center)
    sum_distances += distance_new_center
    probabilities = distances / sum_distances
    c = X[np.random.choice(choice_array, p=probabilities)]
    # TODO: Remove `new_center' from X.
    np.append(centers, c)