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
train_data = pd.read_csv(trainDataAddress)
test_data = pd.read_csv(r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\SourceData\TestData.csv")
y_test = test_data.Subtype

X_train = train_data.iloc[:,1:-2]
y_train = train_data.Subtype
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
le = LabelEncoder()
y = pd.DataFrame(le.fit_transform(y_train))

minMaxScaler = MinMaxScaler()
X_train = pd.DataFrame(minMaxScaler.fit_transform(X_train), index=X_train.index,columns=X_train.columns)

test_data = test_data[X_train.columns]
test_data = pd.DataFrame(minMaxScaler.fit_transform(test_data), index=test_data.index,columns=test_data.columns)


result = {}
names = []
scoring = 'accuracy'
for name, model in models:
    preds = np.zeros(test_data.shape[0])
    kf = KFold(n_splits=5,random_state=48,shuffle=True)
    for trn_idx, test_idx in kf.split(X_train,y):
        X_tr,X_val=X_train.iloc[trn_idx],X_train.iloc[test_idx]
        y_tr,y_val=y.iloc[trn_idx],y.iloc[test_idx]      
        if name == 'XGBM':  
            model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)], eval_metric="auc",early_stopping_rounds=100,verbose=False)
        else:
            model.fit(X_tr, y_tr)
    ty = le.transform(y_test)
    pred = model.predict(test_data)
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