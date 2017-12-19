import numpy as np

USER_FEATURE_DIM = 6
# TODO : fine-tune delta
DELTA = 0.05
ALPHA = 1.0 + np.sqrt(np.log(2.0 / DELTA) / 2.0) 

A = []
b = []
last_user_features = []

def set_articles(articles):

    for article, features in articles.items():
    	#print article, features
    	#so that I don't say I don't do anything in this function..
    	# what to do with the articles features?
    	A.append(np.eye(USER_FEATURE_DIM))
    	b.append(0)

    A = np.array(A)
    b = np.array(b)

# how to do the update without the z_t.
# it's just a hack for now to save the last point's features, to be
# changed.
def update(reward):
	A = A + last_user_features * np.transpose(last_user_features)	
	b = b + reward * last_user_features 

 
def recommend(time, user_features, choices):
	num_actions = choices.shape[0]
	w_hat = np.zeros(num_actions)
	UCB = np.zeros(num_actions) 
	for x in choices:
		print x
		A_x_inv =  np.linalg.inv(A[x])
		w_hat[x] =  A_x_inv * b[x]
		UCB[x] = np.transpose(w_hat[x]) * user_features + ALPHA * np.sqrt(np.transpose(user_features) * A_x_inv * user_features)

	last_user_features = user_features		 
	return np.argmax(UCB)

    #return np.random.choice(choices)
