import sys

from Model.CustomXGBoost import CustomXGBoost
sys.path.insert(0, r'C:\Users\FIRAT.KURT\Documents\Thesis_2021\src')
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
    
    def __init__(_self, XTrain, yTrain):
        estimators = []
        model1 = LinearDiscriminantAnalysis()
        estimators.append(('LDA', model1))
        model2 = LGBMClassifier()
        estimators.append(('LGBM', model2))
        model3 = CustomXGBoost(XTrain, yTrain)
        #model3 = XGBClassifier(learning_rate = 0.04586090618716276, reg_lambda = 0.06826522569951803, reg_alpha = 1.7871177682650604e-06, subsample = 0.40807207936359097, colsample_bytree = 0.3135487605486668, max_depth = 7)
        estimators.append(('XGBM', model3))
        model4 = SVC()
        estimators.append(('SVM', model4))
        model5 = KNeighborsClassifier(n_neighbors=3)
        estimators.append(('KNN', model5))
        # create the ensemble model
        _self.model = VotingClassifier(estimators)
    
    def fit(_self, X, y):
        #for X_tr, X_val, y_tr, y_val, _ in DataManager.GetKFold(X,y):
        #    _self.model.fit(X_tr, y_tr)
        _self.model.fit(X,y)    
        
    def predict(_self, XTest):
        return _self.model.predict(XTest)

    def evals_result(_self):
        return _self.model.evals_result()