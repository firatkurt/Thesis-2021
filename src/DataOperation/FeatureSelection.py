from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

class FeatureSelection:


    def __init__ (self, X, y):
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
    

    def GetviaRFE(self, n):
        rfc = RandomForestClassifier(n_estimators=n)
        rfe = RFE(rfc, n_features_to_select=n)
        fit = rfe.fit(self.X, self.y)
        recursive_features = fit.transform(self.X)
        return recursive_features


    def GetViaFeatureSelection(self, n):
        rfc = RandomForestClassifier(n_estimators=n)
        select_model = SelectFromModel(rfc)
        fit = select_model.fit(self.X, self.y)
        model_features = fit.transform(self.X)
        return model_features

    
    
