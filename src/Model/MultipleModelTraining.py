import sys

import pandas as pd

sys.path.insert(0, r'C:\Users\FIRAT.KURT\PycharmProjects\Thesis-2021\src')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from DataOperation.DataManager import DataManager   
from Model.CustomXGBoost import CustomXGBoost
from Model.CustomLGBMClassifier import CustomLGBMClassifier
from Model.BlendingModel import *
from HyperParameterTune import KNNTuner
from HyperParameterTune import SVCTuner
from Training.TrainingScore import TrainingScore
import itertools as it
from lightgbm import LGBMClassifier
from HyperParameterTune import LGBMClassifierTuner as lgbmt
from HyperParameterTune import XGBoostTuner as xbt
from sklearn.model_selection import train_test_split

trainDataAddress = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TrainDatas\FeatureSelection_20.csv"
testDataAddress  = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TrainDatas\FeatureSelection_20.csv"

def train(trainDataAddress, testDataAddress, numericColumnEncoderName='MinMaxScaler'):
    dm = DataManager.fromCsvFile(trainDataAddress, testDataAddress, numericColumnEncoderName = numericColumnEncoderName,
                                 numericColumns = 'All', columns = (1,-1), label = 'Subtype', encodeLabel = True)
    X,y = dm.GetTrainData()
    X_test, y_test = dm.GetTestData()
    allEstimators = []
    allEstimators.append(('LDA', LinearDiscriminantAnalysis()))
    lgbmParameters = getLGBMTunedParameters(X,y)
    allEstimators.append(('CustomLGBM', CustomLGBMClassifier(**lgbmParameters)))
    xgBoostParameters = getXGBoostTunedParameters(X,y)
    allEstimators.append(('CustomXGB', CustomXGBoost(**xgBoostParameters)))
    svc_c = SVCTuner.hyperParameterTune(X,y)
    allEstimators.append(('SVM', SVC(C=svc_c)))
    k = KNNTuner.hyperParameterTune(X,y)
    allEstimators.append(('KNN', KNeighborsClassifier(n_neighbors=k)))
    allEstimators.append(('AdaBoost', AdaBoostClassifier()))
    estimatorsCombinations = it.combinations(allEstimators, 5)

    models = []
    for estimators in estimatorsCombinations:
        model = EnsambleModel(estimators)
        models.append((model.__str__(), model))
    blender = fit_ensemble(models, X, y)
    yhat = predict_ensemble(models, blender, X_test)
    result = TrainingScore()
    result.accuracy_score = accuracy_score(y_test, yhat)
    result.precision_score = precision_score(y_test, yhat, average='macro')
    result.recall_score = recall_score(y_test, yhat, average='macro')
    result.confusion_matrix = confusion_matrix(y_test, yhat)
    return result

def getLGBMTunedParameters(X, y):
    XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2)
    parameters = lgbmt.hyperParameterTune(XTrain, XValid, yTrain, yValid)
    return parameters

def getXGBoostTunedParameters(X, y):
    XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2)
    parameters = xbt.hyperParameterTune(XTrain, XValid, yTrain, yValid)
    return parameters