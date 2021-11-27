import sys
sys.path.insert(0, r'C:\Users\FIRAT.KURT\PycharmProjects\Thesis-2021\src')
from sklearn.base import BaseEstimator, ClassifierMixin
from lightgbm import LGBMClassifier as lgbm
from DataOperation.DataManager import *

class CustomLGBMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, objective:str = None,
                 eval_metric=None, **parameters: None):
        self.eval_metric = eval_metric
        self.objective = objective
        self.parameters = parameters
        if self.parameters:
            self.model = lgbm(**self.parameters)
        else:
            self.model = lgbm()

    def fit(self, X, y):
        for X_tr, X_val, y_tr, y_val, _ in GetKFold(X, y):
            self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=self.eval_metric)
        return self.model

    def predict(self, data, raw_score=False, start_iteration=0, num_iteration=None,
                pred_leaf=False, pred_contrib=False, **kwargs):
        return self.model.predict(data, raw_score, start_iteration, num_iteration,
                pred_leaf, pred_contrib, **kwargs)

    def predict_proba(self, data, raw_score=False, start_iteration=0, num_iteration=None,
                      pred_leaf=False, pred_contrib=False, **kwargs):
        return self.model.predict_proba(data, raw_score, start_iteration, num_iteration,
                pred_leaf, pred_contrib, **kwargs)

    def evals_result(self):
        return self.model.evals_result()
