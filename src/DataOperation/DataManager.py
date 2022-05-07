import pandas as pd
import sklearn.preprocessing as pp
from DataOperation import ObjectEncoderFactory as oef
from DataOperation import NumericEncoderFactory as nef
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def GetKFold(X, y, n_split=5, shuffle=True, random_state=42):
    y = pd.DataFrame(y)
    kf = KFold(n_splits=n_split, shuffle=shuffle,
               random_state=random_state)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=X, y=y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        yield X_train, X_valid, y_train, y_valid, fold

class DataManager:
       
    def __init__(self,
                 trainData,
                 testData,
                 objectColumns = None,
                 numericColumns = None,
                 columns='All',
                 objectColumnEncoderName ='OneHotEncoder',
                 numericColumnEncoderName ='MinMaxScaler',
                 label=-1,
                 labelFilter = None,
                 encodeLabel=False
                 ):
        self.objectColumns = objectColumns
        self.numericColumns = numericColumns
        self.objectColumnEncoderName = objectColumnEncoderName
        self.numericColumnEncoderName = numericColumnEncoderName
        self.columns = columns
        self.label = label
        self.labelFilter = labelFilter
        self.encodeLabel = encodeLabel
        self.__isTrainDataFilled = False
        self.trainData = trainData
        self.testData = testData
        self.numerericColumnEncoder = None
        self.objectColumnEncoder = None
        self.labelEncoder = None

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
                    labelFilter = None,
                    encodeLabel=False
                    ):
        trainData = pd.read_csv(trainPath)
        testData = pd.read_csv(testPath)
        #trainData.interpolate
        return cls(trainData,testData, objectColumns, numericColumns, columns,objectColumnEncoderName,
                   numericColumnEncoderName, label, labelFilter, encodeLabel)

    @classmethod 
    def fromExactTrainTestSet(cls,
                    XTrain,
                    yTrain,
                    XTest,
                    yTest):
        trainData = pd.concat(XTrain, yTrain, axis=1)
        testData = pd.concat(XTest, yTest, axis=1)
        return cls(trainData,testData)


    def LoadTrainData(self):
        try:
            self.trainData = pd.read_csv(self.trainPath)
        except:
            print("An exception occured when loading train data")
            raise Exception("An exception occured when loading train data")

    def LoadTestData(self):
        try:
            self.testData = pd.read_csv(self.testPath)
        except:
            print("An exception occured when loading test data")
            raise Exception("An exception occured when loading test data")

    def GetTrainData(self):
        if self.trainData is None:
            self.LoadTrainData()
        self._filterTrainData_()
        X = self._fillX(self.trainData)
        X = self._encodeNumericColumns(X)
        X = self._encodeObjectColumns(X)
        y = self._filly(self.trainData)
        y = self._encodeLabelColumns(y)
        self.__isTrainDataFilled = True
        return X, y

    def GetTestData(self):
        if self.testData is None:
            self.LoadTestData()
        if self.__isTrainDataFilled == False:
            self.GetTrainData()
        self._filterTestData_()
        X = self._fillX(self.testData)
        X = self._encodeNumericColumns(X)
        X = self._encodeObjectColumns(X)
        y = self._filly(self.testData)
        y = self._encodeLabelColumns(y)
        return X, y

    #TODO: use GetKFold method 
    def TrainDataKFold(self, n_split=5, shuffle=True, random_state=42):
        kf = KFold(n_splits=n_split, shuffle=shuffle,
                   random_state=random_state)
        X, y = self.GetTrainData()
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X=X, y=y)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            yield X_train, X_valid, y_train, y_valid, fold

    
    def TrainTestSplit(self, test_size = 0.2):
        X,y = self.GetTrainData()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def _filterTrainData_(self):
        if type(self.label) == int:
            self.label = self.trainData.iloc[:, self.label].name
        if self.labelFilter:
            self.trainData = self.trainData.loc[self.trainData[self.label].isin(self.labelFilter)]

    def _filterTestData_(self):
        if type(self.label) == int:
            self.label = self.testData.iloc[:, self.label].name
        if self.labelFilter:
            self.testData = self.testData.loc[self.testData[self.label].isin(self.labelFilter)]

    def _fillX(self, data):
        if type(self.columns) == str and self.columns == 'All':
            X = data.iloc[:, :-1]
        elif type(self.columns) == list:
            X = data[self.columns]
        elif type(self.columns) == tuple:
            X = data.iloc[:, self.columns[0]: self.columns[1]]
        else:
            raise Exception(f"{self.columns} columns is not exist in data.")
        self.columns = X.columns.tolist()
        return X

    def _filly(self, data):
        if self.label == -1:
            y = data.iloc[:, -1]
        elif self.label in data.columns:
            y = data[self.label]
        else:
            raise Exception(f"{self.label} column is not exist in data.")
        return y

    def _encodeObjectColumns(self, X):
        if self.objectColumns is None:
            return X
        X_objectColumns = None
        isTrain = False
        if self.objectColumnEncoder is None:
            isTrain = True
            self.objectColumnEncoder = oef.GetEncoder(
                self.objectColumnEncoderName)
        if self.objectColumns == 'All' :
            X_objectColumns = X.columns
        elif type(self.objectColumns) == list:
            X_objectColumns = X[self.objectColumns]
        elif type(self.objectColumns) == tuple:
            X_objectColumns = X.iloc[:,
                              self.objectColumns[0], self.objectColumns[1]]
        if isTrain:
            X_object = pd.DataFrame(
                self.objectColumnEncoder.fit_transform(X_objectColumns))
        else:
            X_object = pd.DataFrame(
                self.objectColumnEncoder.transform(X_objectColumns))
        X_object.index = X.index
        trainRemoved = X.drop(X_objectColumns.columns, axis=1)
        X = pd.concat([trainRemoved, X_object], axis=1)
        return X

    def _encodeNumericColumns(self, X):
        if self.numericColumns is None:
            return X
        isTrain = False
        if self.numerericColumnEncoder is None:
            isTrain = True
            self.numerericColumnEncoder = nef.GetEncoder(
                self.numericColumnEncoderName)
        X_numericColumns = None
        if self.numericColumns == 'All':
            X_numericColumns = X
        elif type(self.numericColumns) == list:
            X_numericColumns = X[self.numericColumns]
        elif type(self.numericColumns) == tuple:
            X_numericColumns = X.iloc[:,
                               self.numericColumns[0]: self.numericColumns[1]]
        if isTrain:
            X[X_numericColumns.columns] = pd.DataFrame(
                self.numerericColumnEncoder.fit_transform(X_numericColumns), index=X.index, columns=X.columns)
        else:
            X[X_numericColumns.columns] = pd.DataFrame(
                self.numerericColumnEncoder.transform(X_numericColumns), index=X.index, columns=X.columns)
        return X

    def _encodeLabelColumns(self, y):
        if self.encodeLabel == False:
            return y
        if self.labelEncoder is None:
            self.labelEncoder = pp.LabelEncoder()
            return pd.DataFrame(self.labelEncoder.fit_transform(y), columns=[self.label])
        else:
            return pd.DataFrame(self.labelEncoder.transform(y), columns=[self.label])
