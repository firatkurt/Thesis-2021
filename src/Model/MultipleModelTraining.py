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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def train(train_X, train_y, test_X, test_y, le):
    allEstimators = []
    allEstimators.append(('LDA', LinearDiscriminantAnalysis()))
    lgbmParameters = lgbmt.hyperParameterTune(train_X,train_y)
    allEstimators.append(('CustomLGBM', CustomLGBMClassifier(lgbmParameters)))
    xgBoostParameters = xbt.hyperParameterTune(train_X,train_y)
    allEstimators.append(('CustomXGB', CustomXGBoost(xgBoostParameters)))
    svc_c = SVCTuner.hyperParameterTune(train_X,train_y)
    allEstimators.append(('SVM', SVC(C=svc_c, probability=True)))
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
    y_pred = predict_proba_ensemble(models,blender,test_X)
    drawRocCurve(test_y, y_pred, le)
    result = TrainingScore()
    result.accuracy_score = accuracy_score(test_y, yhat)
    result.precision_score = precision_score(test_y, yhat, average='macro')
    result.recall_score = recall_score(test_y, yhat, average='macro')
    result.confusion_matrix = confusion_matrix(test_y, yhat)
    return result

def drawRocCurve(y_true, y_probas, le):
    classes = [0, 1, 2, 3, 4]
    y_true = label_binarize(y_true, classes=classes)
    n_classes = y_true.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    reverseClasses = le.inverse_transform(classes)
    plt.plot(fpr[0], tpr[0], color="b", lw=lw,
             label="%s ROC curve (area = %0.2f)" % (reverseClasses[0], roc_auc[0]), )
    plt.plot(fpr[1], tpr[1], color="g", lw=lw,
             label="%s ROC curve (area = %0.2f)" % (reverseClasses[1], roc_auc[1]), )
    plt.plot(fpr[2], tpr[2], color="c", lw=lw,
             label="%s ROC curve (area = %0.2f)" % (reverseClasses[2], roc_auc[2]), )
    plt.plot(fpr[3], tpr[3], color="r", lw=lw,
             label="%s ROC curve (area = %0.2f)" % (reverseClasses[3], roc_auc[3]), )
    plt.plot(fpr[4], tpr[4], color="y", lw=lw,
             label="%s ROC curve (area = %0.2f)" % (reverseClasses[4], roc_auc[4]), )
    plt.plot(fpr["micro"], tpr["micro"], color="darkorange", lw=lw,
             label="ROC curve (area = %0.2f)" % roc_auc["micro"], )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()