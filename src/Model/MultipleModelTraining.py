import sys
sys.path.insert(0, r'C:\Users\FIRAT.KURT\Documents\Thesis_2021\src')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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

estimators = []
model1 = LinearDiscriminantAnalysis()
estimators.append(('LDA', model1))
model2 = LGBMClassifier()
estimators.append(('LGBM', model2))
model3 = XGBClassifier(learning_rate = 0.04586090618716276, reg_lambda = 0.06826522569951803, reg_alpha = 1.7871177682650604e-06, subsample = 0.40807207936359097, colsample_bytree = 0.3135487605486668, max_depth = 7)
estimators.append(('XGBM', model3))
model4 = SVC()
estimators.append(('SVM', model4))
model5 = KNeighborsClassifier(n_neighbors=3)
estimators.append(('KNN', model5))
# create the ensemble model
ensemble = VotingClassifier(estimators)


models = []
#models.append(('LR', LogisticRegression()))
models.append(('ensemble', VotingClassifier(estimators)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LGBM', LGBMClassifier()))
models.append(('XGBM', XGBClassifier(objective="multi:softmax", learning_rate = 0.04586090618716276, reg_lambda = 0.06826522569951803, reg_alpha = 1.7871177682650604e-06, subsample = 0.40807207936359097, colsample_bytree = 0.3135487605486668, max_depth = 7)))
models.append(('SVM', SVC()))
models.append(('KNN', model5))



trainDataAddress = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TrainDatas\FeatureSelection_20.csv"
testDataAddress  = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\SourceData\TestData.csv"

dm = DataManager(trainDataAddress, testDataAddress, numericColumns = 'All', columns = (1,-2), label = 'Subtype', encodeLabel = True)


result = {}
names = []
scoring = 'accuracy'
for name, model in models:
    for X_tr, X_val, y_tr, y_val, _ in dm.TrainDataKFold():
        #if name == 'XGBM':  
        #    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)], eval_metric="auc",early_stopping_rounds=100,verbose=False)
        #else:
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