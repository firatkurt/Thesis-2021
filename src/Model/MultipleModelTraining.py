import sys
sys.path.insert(0, r'..\\')

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
from HyperParameterTune import LGBMClassifierTuner as lgbmt
from HyperParameterTune import XGBoostTuner as xbt

def train(trainDataAddress, testDataAddress, numericColumnEncoderName='MinMaxScaler', labelFilter=None):
    dm = DataManager.fromCsvFile(trainDataAddress, testDataAddress, numericColumnEncoderName=numericColumnEncoderName,
                                 numericColumns='All',
                                 label='Subtype', labelFilter=labelFilter, encodeLabel=True)
    X,y = dm.GetTrainData()
    X_test, y_test = dm.GetTestData()
    allEstimators = []
    allEstimators.append(('LDA', LinearDiscriminantAnalysis()))
    lgbmParameters = lgbmt.hyperParameterTune(X,y)
    allEstimators.append(('CustomLGBM', CustomLGBMClassifier(**lgbmParameters)))
    xgBoostParameters = xbt.hyperParameterTune(X,y)
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