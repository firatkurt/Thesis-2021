
import sys

import numpy as np

sys.path.insert(0, r'..//')
from numpy import hstack
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from DataOperation.DataManager import DataManager
from Model.CustomXGBoost import CustomXGBoost
from Model.EnsambleModel import EnsambleModel
from sklearn.model_selection import KFold


# fit the blending ensemble
def fit_ensemble(models, X, y):
	# fit all models on the training set and predict on hold out set
	meta_X = np.zeros([X.shape[0], len(models)])
	kf = KFold(n_splits=5, shuffle=True,random_state=42)

	for index, (name, model) in enumerate(models):
		for fold, (train_idx, valid_idx) in enumerate(kf.split(X=X, y=y)):
			X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
			y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]
			# fit in training set
			model.fit(X_train, y_train)
			# predict on hold out set
			yhat = model.predict(X_val)
			# reshape predictions into a matrix with one column
			#yhat = yhat.reshape(len(yhat), 1)
			# store predictions as input for blending
			#meta_X.append(yhat)
			meta_X[valid_idx,  index] = yhat
	# create 2d array from predictions, each set is an input feature
	#meta_X = hstack(meta_X)
	# define blending model
	blender = RandomForestClassifier(n_estimators=1000, max_depth=5)
	# fit on predictions from base models
	blender.fit(meta_X, y)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for name, model in models:
		# predict with base model
		yhat = model.predict(X_test)
		# reshape predictions into a matrix with one column
		yhat = yhat.reshape(len(yhat), 1)
		# store prediction
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)


def predict_proba_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for name, model in models:
		# predict with base model
		yhat = model.predict(X_test)
		# store prediction
		yhat = yhat.reshape(len(yhat), 1)
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict_proba(meta_X)
