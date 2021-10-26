import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


featureList = ['ABCA10', 'ADAMTS5', 'ANXA1', 'MAGI2-AS3', 'PAMR1', 'SCN4B', 'SPRY2', 'TMEM220', 'TSLP', 'VEGFD']


trainData = pd.read_csv("C:\\Users\\FIRAT.KURT\\Documents\\Thesis_2021\\TrainData.csv")
X_train = trainData[featureList]
y_train = trainData.Subtype
le = LabelEncoder()
y = le.fit_transform(y_train)
minMaxScaler = MinMaxScaler()
X_train = pd.DataFrame(minMaxScaler.fit_transform(X_train), index=X_train.index,columns=X_train.columns)

trainX, validX, trainy, validy  = train_test_split(X_train, y_train, train_size=0.8, random_state = 0)



model = XGBClassifier()

model.fit(trainX,trainy)

preds = model.predict(validX)

accuracy_score(validy, preds)