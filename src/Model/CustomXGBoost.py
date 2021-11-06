import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from DataOperation.DataManager import DataManager
import EvalMetricFactory as emf


class CustomXGBoost:
    
    def __init__(_self, dataModel, objective="multi:softmax", n_estimators=100):
        _self.model = XGBClassifier(
            objective=objective, n_estimators=n_estimators)
        _self.dm = dataModel

    def __init__(_self, dataModel, objective="multi:softmax", n_estimator=100, **parameters):
        _self.model = XGBClassifier(
            objective=objective, n_estimators=n_estimator, **parameters)
        _self.dm = dataModel

    @classmethod
    def FromExactData(cls, XTrain, yTrain, XTest, yTest, objective="multi:softmax", n_estimators=100):
        dataModel = DataManager.fromExactTrainTestSet(XTrain, yTrain, XTest, yTest)
        return cls(dataModel, objective, n_estimators)
    
    @classmethod
    def FromExactData(cls, XTrain, yTrain, XTest, yTest, objective="multi:softmax", n_estimators=100, **parameters):
        dataModel = DataManager.fromExactTrainTestSet(XTrain, yTrain, XTest, yTest)
        return cls(dataModel, objective, n_estimators,**parameters)
    
    def fit(_self):
        for X_tr, X_val, y_tr, y_val, _ in _self.dm.TrainDataKFold():
            _self.model.fit(X_tr, y_tr, eval_set=[
                            (X_val, y_val)], eval_metric="auc", early_stopping_rounds=100, verbose=False)
    
    def transform(_self):
        return _self.model.trasform(_self.dm.GetTrainData())


    def fit_transform(_self):
        _self.fit()
        return _self.trasform()

    def predict(_self):
        Xtest, _ = _self.dm.GetTestData()
        return _self.model.predict(Xtest)

    def eval(_self, eval_metric_name):
        _, y_true = _self.dm.GetTestData()
        pred = emf.GetEvalMetric(
            eval_metric_name, y_true=y_true, y_pred=_self.predict())
        return pred
