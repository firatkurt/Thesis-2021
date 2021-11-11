import sys
sys.path.insert(0, r'C:\Users\FIRAT.KURT\PycharmProjects\Thesis-2021\src')
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from DataOperation.DataManager import DataManager   
from Model.CustomXGBoost import CustomXGBoost
from Model.EnsambleModel import EnsambleModel
from Model.BlendingModel import *

trainDataAddress = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TrainDatas\FeatureSelection_20.csv"
testDataAddress  = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TrainDatas\FeatureSelection_20.csv"#r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\SourceData\TestData.csv"

dm = DataManager.fromCsvFile(trainDataAddress, testDataAddress, numericColumns = 'All', columns = (1,-2), label = 'Subtype', encodeLabel = True)

X,y = dm.GetTrainData()
X_test, y_test = dm.GetTestData()
models = []
models.append(('ensemble', EnsambleModel(X, y)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LGBM', LGBMClassifier()))
models.append(('CustomXGB', CustomXGBoost.initwithtune(X, y)))
models.append(('XGBM', CustomXGBoost(objective= "multi:softmax",
                                     parameters = {"learning_rate": 0.04586090618716276, 
                                                    "reg_lambda" : 0.06826522569951803, 
                                                    "reg_alpha" :1.7871177682650604e-06,
                                                    "subsample" : 0.40807207936359097, 
                                                    "colsample_bytree" : 0.3135487605486668, 
                                                    "max_depth" :7 })))
models.append(('SVM', SVC()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
# summarize data split
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))

# train the blending ensemble
blender = fit_ensemble(models, X, y)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Blending Accuracy: %.3f' % (score * 100))

result = {}
names = []
scoring = 'accuracy'
for name, model in models:
    for X_tr, X_val, y_tr, y_val, _ in dm.TrainDataKFold():
        model.fit(X_tr, y_tr)
    X_test, ty = dm.GetTestData()
    pred = model.predict(X_test)
    print(name)
    print(pred)
    acc = accuracy_score(ty,pred)
    print(acc)
    prec = precision_score(ty,pred, average='macro')
    print(prec)
    rec = recall_score(ty,pred, average='macro')
    print(rec)
    con = confusion_matrix(ty,pred)
    print(con)
    result[name] = ( "Accuracy: " + str(acc), "Precision:" + str(prec),
                     "Recall: " + str(rec), "con:" + str(con))



