import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

np.random.seed(0)

D = 400
NEW_D = 30000
INDICES = np.random.choice(np.arange((D * (D + 1)) // 2), size=NEW_D)

def transform(X):
    # make sure this function works with both 1D (including Python lists) and
    # 2D arrays
    if type(X) == list:
        X = np.array(X)
    if X.ndim == 1:
        return np.outer(X, X).flatten()

    n = X.shape[0]
    new_dim = X.shape[1] + X.shape[1]**2
    Z = np.ndarray((n, new_dim))

    for i, x in enumerate(X):
        if i % 100 == 0:
            print i
        Z[i, :] = np.append(x, np.outer(x, x).flatten())

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

print 'File loaded!'

X = np.array(X)
Y = np.array(Y)

X = transform(X)
print 'Data transformed!'

feature_selector = SelectKBest(score_func=f_classif, k=30000)
feature_selector.fit(X, Y)
X = feature_selector.transform(X)
features = feature_selector.get_support(True)

f = open('data/features.txt', 'w')

f.write('INDICES = [')
for i, feature in enumerate(features):
    if i==0:
        f.write(str(feature))
    else:
        f.write(', ' + str(feature))