from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import FeatureSelection
import pandas as pd
import numpy as nm


class FeatureSelection:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def GetViaSelectKBest(self, n):
        self.k_best = SelectKBest(score_func=f_classif, k=n)
        fit = self.k_best.fit(self.X, self.y)
        univariate_features = fit.transform(self.X)
        mask = self.k_best.get_support()
        new_features = []
        for bool, feature in zip(mask, self.X.columns):
            if bool:
                new_features.append(feature)

        return new_features

    def GetViaRFE(self, n):
        rfc = RandomForestClassifier(n_estimators=100)
        rfe = RFE(rfc, n_features_to_select=n)
        model = rfe.fit(self.X, self.y)
        model.transform(self.X)
        feature_idx = model.get_support()
        feature_name = X.columns[feature_idx]
        return feature_name

    def GetViaFeatureSelection(self, n):
        rfc = RandomForestClassifier(n_estimators=100)
        select_model = SelectFromModel(rfc, max_features=n)
        model = select_model.fit(self.X, self.y)
        model.transform(self.X)
        feature_idx = model.get_support()
        feature_name = X.columns[feature_idx]
        return feature_name
    
    
df = pd.read_csv(r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\SourceData\METABRIC-T.csv", header=1, sep=";", decimal=',')
print(len(df.values))
print(len(df.columns))
#df = df.T
#df.to_csv(r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\SourceData\METABRIC-T.csv", index=False, sep=';')

X = df.iloc[:,2:]
y = df.iloc[:,0]

feature_counts = (20,25,30,50,100)
feature_dict = {}
fs = FeatureSelection(X,y)
for i in feature_counts:
    features = fs.GetViaSelectKBest(i)
    feature_dict["SelectKBest_" + str(i)] = features
    features = fs.GetViaFeatureSelection(i)
    feature_dict["FeatureSelection_" + str(i)] = features
print("KBest and FeatureSelection Completed")
fs = FeatureSelection(X,y)
features = fs.GetViaFeatureSelection(750)
X = X[features]
fs = FeatureSelection(X,y)
for i in feature_counts:
    features = fs.GetViaRFE(i)
    feature_dict["RFE_" + str(i)] = features

for k,v in feature_dict.items():
    selectedData = df[v]
    result = pd.concat([selectedData, y], axis=1)
    result.to_csv(r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\MetaBrickData\\" + k + ".csv",index=False)