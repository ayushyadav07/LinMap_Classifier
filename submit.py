import numpy as np
from sklearn.svm import LinearSVC
from scipy import linalg
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression


def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    n = np.shape(X)[1]
    X = np.cumprod(np.flip(2 * X - 1, axis=1), axis=1)
    feat = linalg.khatri_rao(X.T, X.T).T
    upp_mat = np.triu(np.ones((n, n)), k=0)
    u1d = upp_mat.flatten().astype(bool)
    feat = feat[:,u1d]
    return feat

def my_fit(features, labels):
    # Extract feature data and labels from train_CRPs
    # Fit a logistic regression model
    C=1.0
    features = my_map(features)
    print(np.shape(features))
    model = LogisticRegression(C=C)
    model.fit(features, labels)
    W = model.coef_[0]
    b = model.intercept_[0]
    return W, b

def predict(model, test_challenges):
    # Map test challenges to features and predict responses
    features = my_map(test_challenges)
    return model.predict(features)