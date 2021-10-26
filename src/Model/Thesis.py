import numpy as nm;
import pandas as pd
from pandas.io import html
from pandas_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import csv

#df = pd.read_excel("C:\\Users\\FIRAT.KURT\\Documents\\Thesis_2021\\sample.xlsx")
trainData = pd.read_csv("C:\\Users\\FIRAT.KURT\\Documents\\Thesis_2021\\TrainData.csv")
X_train = trainData.iloc[:,1:-1]
y_train = trainData.Subtype

print(X_train.isnull().any().any())
print(X_train.columns)

minMaxScaler = MinMaxScaler()
X_train = pd.DataFrame(minMaxScaler.fit_transform(X_train), index=X_train.index,columns=X_train.columns)

feature_List = [10,20,50,100,200,500,750]
result = {}
for i in feature_List:
    k_best = SelectKBest(score_func=f_classif, k=i)
    fit = k_best.fit(X_train, y_train)
    univariate_features = fit.transform(X_train)

    mask = k_best.get_support() #list of booleans
    new_features = [] # The list of your K best features

    for bool, feature in zip(mask, X_train.columns):
        if bool:
            new_features.append(feature)

    result[i] = new_features

with open('FeatureList.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in result.items():
       writer.writerow([key, value])
