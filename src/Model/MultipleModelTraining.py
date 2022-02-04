import sys
sys.path.insert(0, r'..\\')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from DataOperation.DataManager import DataManager   
from Model.CustomXGBoost import CustomXGBoost
from Model.CustomLGBMClassifier import CustomLGBMClassifier
from Model.BlendingModel import *
from HyperParameterTune import KNNTuner
from HyperParameterTune import SVCTuner
from Training.TrainingScore import TrainingScore
import itertools as it
from HyperParameterTune import LGBMClassifierTuner as lgbmt
from HyperParameterTune import XGBoostTuner as xbt


def train(train_X, train_y, test_X, test_y):
    allEstimators = []
    allEstimators.append(('LDA', LinearDiscriminantAnalysis()))
    lgbmParameters = lgbmt.hyperParameterTune(train_X,train_y)
    allEstimators.append(('CustomLGBM', CustomLGBMClassifier(**lgbmParameters)))
    xgBoostParameters = xbt.hyperParameterTune(train_X,train_y)
    allEstimators.append(('CustomXGB', CustomXGBoost(**xgBoostParameters)))
    svc_c = SVCTuner.hyperParameterTune(train_X,train_y)
    allEstimators.append(('SVM', SVC(C=svc_c)))
    k = KNNTuner.hyperParameterTune(train_X,train_y)
    allEstimators.append(('KNN', KNeighborsClassifier(n_neighbors=k)))
    allEstimators.append(('AdaBoost', AdaBoostClassifier()))
    estimatorsCombinations = it.combinations(allEstimators, 5)

    models = []
    for estimators in estimatorsCombinations:
        model = EnsambleModel(estimators)
        models.append((model.__str__(), model))
    blender = fit_ensemble(models, train_X,train_y)
    yhat = predict_ensemble(models, blender, test_X)
    drawRocCurve(test_y,yhat)
    result = TrainingScore()
    result.accuracy_score = accuracy_score(test_y, yhat)
    result.precision_score = precision_score(test_y, yhat, average='macro')
    result.recall_score = recall_score(test_y, yhat, average='macro')
    result.confusion_matrix = confusion_matrix(test_y, yhat)
    return result

def drawRocCurve(y_true, y_probas):
    import scikitplot as skplt
    import matplotlib.pyplot as plt


    skplt.metrics.plot_roc_curve(y_true, y_probas)
    plt.show()