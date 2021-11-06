
import sys
from numpy.lib.function_base import average
sys.path.insert(0, r'C:\Users\FIRAT.KURT\Documents\Thesis_2021\src')
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from DataOperation.DataManager import DataManager
from HyperParameterTune import XGBoostTuner as xbt
import EvalMetricFactory as emf
from sklearn.model_selection import train_test_split


class CustomXGBoost:

    def __init__(_self, eval_metric="auc",  early_stopping_rounds=100, n_estimator=100, **parameters):
        _self.eval_metric = eval_metric
        _self.early_stopping_rounds = early_stopping_rounds
        _self.model = XGBClassifier(
            objective = "multi:softmax", n_estimators=n_estimator, **parameters)
    
    @classmethod
    def InitWithTune(cls, X, y, eval_metric="auc",  early_stopping_rounds=100, n_estimator=100):
        XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size = 0.2)
        parameters = xbt.hyperParameterTune(XTrain, XValid, yTrain, yValid)
        return cls(eval_metric, early_stopping_rounds, n_estimator, parameters)


    def fit(_self, X, y):
        for X_tr, X_val, y_tr, y_val, _ in DataManager.GetKFold(X,y):
            _self.model.fit(X_tr, y_tr, eval_set=[
                            (X_val, y_val)], eval_metric=_self.eval_metric, early_stopping_rounds=_self.early_stopping_rounds, verbose=False)
    
    def predict(_self, XTest):
        return _self.model.predict(XTest)

    def evals_result(_self):
        return _self.model.evals_result()
