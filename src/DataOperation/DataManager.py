import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
from DataOperation import ObjectEncoderFactory as oef
from DataOperation import NumericEncoderFactory as nef
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class DataManager:
       
    def __init__(_self,
                 trainData,
                 testData,
                 objectColumns = None,
                 numericColumns = None,
                 columns='All',
                 objectColumnEncoderName ='OneHotEncoder',
                 numericColumnEncoderName ='MinMaxScaler',
                 label=-1,
                 encodeLabel=False
                 ):
        _self.objectColumns = objectColumns
        _self.numericColumns = numericColumns
        _self.objectColumnEncoderName = objectColumnEncoderName
        _self.numericColumnEncoderName = numericColumnEncoderName
        _self.columns = columns
        _self.label = label
        _self.encodeLabel = encodeLabel
        _self.__isTrainDataFilled = False
        _self.trainData = trainData
        _self.testData = testData
        _self.numerericColumnEncoder = None
        _self.objectColumnEncoder = None
        _self.labelEncoder = None

    @classmethod 
    def fromCsvFile(cls,
                    trainPath,
                    testPath,
                    objectColumns = None,
                    numericColumns = None,
                    columns='All',
                    objectColumnEncoderName ='OneHotEncoder',
                    numericColumnEncoderName ='MinMaxScaler',
                    label=-1,
                    encodeLabel=False
                    ):
        trainData = pd.read_csv(trainPath)
        testData = pd.read_csv(testPath)
        return cls(trainData,testData, objectColumns, numericColumns, columns,objectColumnEncoderName, numericColumnEncoderName, label, encodeLabel)

    @classmethod 
    def fromExactTrainTestSet(cls,
                    XTrain,
                    yTrain,
                    XTest,
                    yTest):
        trainData = pd.concat(XTrain, yTrain, axis=1)
        testData = pd.concat(XTest, yTest, axis=1)
        return cls(trainData,testData)


    def LoadTrainData(_self):
        try:
            _self.trainData = pd.read_csv(_self.trainPath)
        except:
            print("An exception occured when loading train data")
            raise Exception("An exception occured when loading train data")

    def LoadTestData(_self):
        try:
            _self.testData = pd.read_csv(_self.testPath)
        except:
            print("An exception occured when loading test data")
            raise Exception("An exception occured when loading test data")

    def GetTrainData(_self):
        if _self.trainData is None:
            _self.LoadTrainData()
        X = _self._fillX(_self.trainData)
        X = _self._encodeNumericColumns(X)
        X = _self._encodeObjectColumns(X)
        y = _self._filly(_self.trainData)
        y = _self._encodeLabelColumns(y)
        _self.__isTrainDataFilled = True
        return X, y

    def GetTestData(_self):
        if _self.testData is None:
            _self.LoadTestData()
        if _self.__isTrainDataFilled == False:
            _self.GetTrainData()
        X = _self._fillX(_self.testData)
        X = _self._encodeNumericColumns(X)
        X = _self._encodeObjectColumns(X)
        y = _self._filly(_self.testData)
        y = _self._encodeLabelColumns(y)
        return X, y

    def TrainDataKFold(_self, n_split=5, shuffle=True, random_state=42):
        kf = KFold(n_splits=n_split, shuffle=shuffle,
                   random_state=random_state)
        X, y = _self.GetTrainData()
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X=X, y=y)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            yield X_train, X_valid, y_train, y_valid, fold

    def TrainTestSplit(_self, test_size = 0.2):
        X,y = _self.GetTrainData()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def _fillX(_self, data):
        if type(_self.columns) == str and _self.columns == 'All':
            X = data.iloc[:, :-1]
        elif type(_self.columns) == list:
            X = data[_self.columns]
        elif type(_self.columns) == tuple:
            X = data.iloc[:, _self.columns[0] : _self.columns[1]]
        else:
            raise Exception(f"{_self.columns} columns is not exist in data.")
        _self.columns = X.columns.tolist()
        return X

    def _filly(_self, data):
        if _self.label == -1:
            y = data.iloc[:, -1]
        elif _self.label in data.columns:
            y = data[_self.label]
        else:
            raise Exception(f"{_self.label} column is not exist in data.")
        return y

    def _encodeObjectColumns(_self, X):
        if _self.objectColumns is None:
            return X
        X_objectColumns = None
        isTrain = False
        if _self.objectColumnEncoder is None:
            isTrain = True
            _self.objectColumnEncoder = oef.GetEncoder(
                _self.objectColumnEncoderName)
        if _self.objectColumns == 'All' :
            X_objectColumns = X.columns
        elif type(_self.objectColumns) == list:
            X_objectColumns = X[_self.objectColumns]
        elif type(_self.objectColumns) == tuple:
            X_objectColumns = X.iloc[:,
                                    _self.objectColumns[0], _self.objectColumns[1]]
        if isTrain:
            X_object = pd.DataFrame(
                _self.objectColumnEncoder.fit_transform(X_objectColumns))
        else:
            X_object = pd.DataFrame(
                _self.objectColumnEncoder.transform(X_objectColumns))
        X_object.index = X.index
        trainRemoved = X.drop(X_objectColumns.columns, axis=1)
        X = pd.concat([trainRemoved, X_object], axis=1)
        return X

    def _encodeNumericColumns(_self, X):
        if _self.numericColumns is None:
            return X
        isTrain = False
        if _self.numerericColumnEncoder is None:
            isTrain = True
            _self.numerericColumnEncoder = nef.GetEncoder(
                _self.numericColumnEncoderName)
        X_numericColumns = None
        if _self.numericColumns == 'All' :
            X_numericColumns = X
        elif type(_self.numericColumns) == list:
            X_numericColumns = X[_self.numericColumns]
        elif type(_self.numericColumns) == tuple:
            X_numericColumns = X.iloc[:,
                                     _self.numericColumns[0] : _self.numericColumns[1]]
        if isTrain:
            X[X_numericColumns.columns] = pd.DataFrame(
                _self.numerericColumnEncoder.fit_transform(X_numericColumns), index=X.index, columns=X.columns)
        else:
            X[X_numericColumns.columns] = pd.DataFrame(
                _self.numerericColumnEncoder.transform(X_numericColumns), index=X.index, columns=X.columns)
        return X

    def _encodeLabelColumns(_self, y):
        if _self.encodeLabel == False:
            return y
        if _self.labelEncoder is None:
            _self.labelEncoder = pp.LabelEncoder()
            return pd.DataFrame(_self.labelEncoder.fit_transform(y))
        else:
            return pd.DataFrame(_self.labelEncoder.transform(y))
