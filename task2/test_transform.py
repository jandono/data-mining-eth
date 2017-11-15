import numpy as np
# from sklearn import *

def transform_poly(X, degree):
	# make sure this function works with both 1D (including Python lists) and
	# 2D arrays
	if type(X) == list:
		X = np.array(X)

	NEW_X = np.ndarray(X.shape[0], dtype=list)  	
	X_poly = np.ndarray((X.shape[0], degree), dtype=np.ndarray)
	for r in range(X.shape[0]):
		print r	
		row = X[r]
		X_poly[r][0] = np.array(row)

		for i in range(1, degree):
			X_poly[r][i] = np.ravel(np.outer(X_poly[r][i-1], row))
	
		NEW_X[r] = np.concatenate(X_poly[r]).tolist()

	return np.array(NEW_X)


def transform(X):
    # make sure this function works with both 1D (including Python lists) and
    # 2D arrays
    if type(X) == list:
        X = np.array(X)
    if X.ndim == 1:
        return np.outer(X, X).flatten()[INDICES]

    n = X.shape[0]
    Z = np.ndarray((n, NEW_D))

    for i, x in enumerate(X):
        Z[i, :] = np.outer(x, x).flatten()[INDICES]

    return Z


f = open('data/handout_train.txt', 'r')

X = [] 
Y = []
for line in f:
	parts = line.split()
	for i in range(len(parts)):
		parts[i] = float(parts[i])
	X.append(parts[1:])
	Y.append(parts[0])

X = np.array(X)
Y = np.array(Y)

transform_poly(X, 2)
print 'Success'

# x = np.array([[1,2,3],[4,5,6]])
# print transform_poly(x, 3)