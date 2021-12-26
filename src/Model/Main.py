import sys
sys.path.insert(0, r'..\\')
#sys.path.insert(0, r'C:\Users\FIRAT.KURT\PycharmProjects\Thesis-2021\src')
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from Model.CustomXGBoost import CustomXGBoost
from HyperParameterTune import SVCTuner

import MultipleModelTraining as mmt
import os
import itertools as it
import csv
import pandas as pd


root = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TrainDatas"
testDataAddress = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\SourceData\TestData.csv"
trainFiles = os.listdir(root)

#trainData = pd.read_csv(os.path.join(root, 'FeatureSelection_20.csv'))
#X = trainData.iloc[:,1:-2]
#y = trainData.iloc[:,-2]

#svcParam = SVCTuner.hyperParameterTune(X, y)
def calculateAllModelResult():
    numericEncoderlist = ['MinMaxScaler','StandardScaler', 'RobustScaler']

    allModels = it.product(*[trainFiles, numericEncoderlist])

    results = {}
    for model in allModels:
        activeTrainingFilePath = os.path.join(root, model[0])
        result = mmt.train(activeTrainingFilePath, testDataAddress, model[1])
        results[model] = result
        print(model[0][:-4] +'-' + model[1] +',' + result.__str__())

    with open('ModelResults.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
           writer.writerow(key[0][:-4] +'-' + key[1] +',' + value.__str__())

def calculateHealtyAndOtherResult(fileName, encoderName, labelFilter):
    activeTrainingFilePath = os.path.join(root,fileName)
    result = mmt.train(activeTrainingFilePath, testDataAddress, labelFilter=labelFilter,
                       numericColumnEncoderName= encoderName)
    return  result


if __name__ ==  '__main__':
    basel = calculateHealtyAndOtherResult('FeatureSelection_20.csv', 'RobustScaler', ['Basal', 'Healty'])
    LumB = calculateHealtyAndOtherResult('FeatureSelection_20.csv','RobustScaler',['LumB', 'Healty'] )
    LumA = calculateHealtyAndOtherResult('FeatureSelection_20.csv','RobustScaler',['LumA', 'Healty'] )
    all = calculateHealtyAndOtherResult('FeatureSelection_20.csv', 'RobustScaler', ['Basal',  'LumB', 'LumA','Healty'])
    print("basel:" + basel.__str__())
    print("LumB:" + LumB.__str__())
    print("LumA:" + LumA.__str__())
    print("LumA:" + all.__str__())