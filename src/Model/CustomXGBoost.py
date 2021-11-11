import sys
sys.path.insert(0, r'C:\Users\FIRAT.KURT\PycharmProjects\Thesis-2021\src')
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from DataOperation.DataManager import *
from HyperParameterTune import XGBoostTuner as xbt
from sklearn.model_selection import train_test_split


class CustomXGBoost(BaseEstimator, ClassifierMixin):

    @classmethod
    def initwithtune(cls, X, y, objective="multi:softmax", n_estimators=100,
                     eval_metric="merror",  early_stopping_rounds=100):
        XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2)
        parameters = xbt.hyperParameterTune(XTrain, XValid, yTrain, yValid)
        return cls(objective, n_estimators, eval_metric, early_stopping_rounds, **parameters)

    def __init__(self, objective:str = "multi:softmax", n_estimators:int = 100,
                 eval_metric="merror", early_stopping_rounds=100, **parameters):
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.model = XGBClassifier(objective=objective, **parameters)

    def fit(self, X, y):
        for X_tr, X_val, y_tr, y_val, _ in GetKFold(X, y):
            self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=self.eval_metric,
                           early_stopping_rounds=self.early_stopping_rounds, verbose=False)
        return self.model

    def predict(self, data, output_margin=False, ntree_limit=None,
                validate_features=True, base_margin=None):
        return self.model.predict(data,output_margin, ntree_limit, validate_features, base_margin)

    def predict_proba(self, data, ntree_limit=None, validate_features=True,
                       base_margin=None):
        return self.model.predict_proba(data, ntree_limit,validate_features,base_margin)

    def evals_result(self):
        return self.model.evals_result()
