import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc
from itertools import cycle


rootDir =  "C:\\Users\\FIRAT.KURT\\Documents\\Thesis_2021\\"
trainDataAddress = "C:\\Users\\FIRAT.KURT\\Documents\\Thesis_2021\\TrainDatas\\"
filelist = os.listdir(trainDataAddress)
testData = pd.read_csv("C:\\Users\\FIRAT.KURT\\Documents\\Thesis_2021\\TestData.csv")
y_test = testData.Subtype

result = {}
for file in filelist:
    train_data = pd.read_csv(os.path.join(trainDataAddress , file))
    X_train = train_data.iloc[:,1:-1]
    y_train = train_data.Subtype
    le = LabelEncoder()
    y = pd.DataFrame(le.fit_transform(y_train))
    le.classes_
    minMaxScaler = MinMaxScaler()
    X_train = pd.DataFrame(minMaxScaler.fit_transform(X_train), index=X_train.index,columns=X_train.columns)

    test_data = testData[X_train.columns]
    test_data = pd.DataFrame(minMaxScaler.fit_transform(test_data), index=test_data.index,columns=test_data.columns)

    Best_trial = {'lambda': 0.004420780435449923, 'alpha': 2.083758739908325, 'colsample_bytree': 0.7, 'subsample': 0.6, 'learning_rate': 0.02, 'max_depth': 15, 'random_state': 48, 'min_child_weight': 28}
    preds = np.zeros(test_data.shape[0])
    kf = KFold(n_splits=5,random_state=48,shuffle=True)
    rmse=[]  # list contains rmse for each fold
    n=0
    model = XGBClassifier(objective="multi:softmax", n_estimators=110, nthread=-1,seed=1729, **Best_trial)
    for trn_idx, test_idx in kf.split(X_train,y):
        X_tr,X_val=X_train.iloc[trn_idx],X_train.iloc[test_idx]
        y_tr,y_val=y.iloc[trn_idx],y.iloc[test_idx]
        
        model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)], eval_metric="auc",early_stopping_rounds=100,verbose=False)
        #preds+=model.predict(test_data)/kf.n_splits
        #rmse.append(accuracy_score(y_val, model.predict(X_val)))
        #print(n+1,rmse[n])
        n+=1
    ty = le.transform(y_test)
    pred = model.predict(test_data)
    print(pred)
    acc = accuracy_score(ty,pred)
    print(acc)
    prec = precision_score(ty,pred, average='macro')
    print(prec)
    rec = recall_score(ty,pred, average='macro')
    print(rec)
    con = confusion_matrix(ty,pred)
    print(con)
    result[file] = ( "Accuracy: " + str(acc), "Precision:" + str(prec),
                     "Recall: " + str(rec), "con:" + str(con))


with open(os.path.join(rootDir,'file.txt'), 'w') as file:
     file.write(json.dumps(result)) 

import csv
with open('result.csv', 'w', newline='') as csvfile:
    fieldnames = ['FileName', 'Proporties']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for key in result:
        writer.writerow({'FileName': key, 'Proporties': result[key]})