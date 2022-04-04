import sys
sys.path.insert(0, r'..//')
from Model.CustomXGBoost import CustomXGBoost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from DataOperation.DataManager import DataManager    

class EnsambleModel:
    
    def __init__(self, estimators):
        estimators = estimators
        self.model = VotingClassifier(estimators, voting='soft')
    
    def fit(self, X, y):
        #for X_tr, X_val, y_tr, y_val, _ in DataManager.GetKFold(X,y):
        #    self.model.fit(X_tr, y_tr)
        self.model.fit(X,y)    
        
    def predict(self, XTest):
        return self.model.predict(XTest)

    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def evals_result(self):
        return self.model.evals_result()

    def __str__(self):
        names, _ = zip(*self.model.estimators)
        result = '_'.join(names)
        return result
