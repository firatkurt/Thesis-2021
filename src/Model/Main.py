import sys
sys.path.insert(0, r'..//')
#sys.path.insert(0, r'..\')
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from Model.CustomXGBoost import CustomXGBoost
from HyperParameterTune import SVCTuner
from sklearn.model_selection import train_test_split
import MultipleModelTraining as mmt
import os
import itertools as it
import csv
import pandas as pd
from DataOperation.DataManager import DataManager
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#root = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TestDatas"
#testDataAddress = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TrainDatas\Intersected20.csv"

#root = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\MetaBrickData"
root = r"/Users/firatkurt/Documents/Thesis_Data/Selected_RFE_50"
trainFiles = os.listdir(root)

#trainData = pd.read_csv(os.path.join(root, 'FeatureSelection_20.csv'))
#X = trainData.iloc[:,1:-2]
#y = trainData.iloc[:,-2]

#svcParam = SVCTuner.hyperParameterTune(X, y)
def calculateAllModelResult():
    numericEncoderlist = ['StandardScaler'] #['MinMaxScaler','StandardScaler', 'RobustScaler']

    allModels = it.product(*[trainFiles, numericEncoderlist])

    results = {}
    for model in allModels:
        activeTrainingFilePath = os.path.join(root, model[0])
        data = pd.read_csv(activeTrainingFilePath, header=0)
        y = data['Subtype']
        trainData, testData = train_test_split(data, test_size=0.2, stratify=y)
        # testData = pd.read_csv(testDataAddress)
        labelFilter = ['Basal', 'LumB', 'LumA', 'Her2', 'Normal']
        dm = DataManager(trainData, testData, numericColumnEncoderName=model[1], numericColumns='All',
                         label='Subtype', labelFilter=labelFilter, encodeLabel=True)
        X, y = dm.GetTrainData()
        test_X, test_y = dm.GetTestData()

        result = mmt.train(X,y,test_X,test_y)
        results[model] = result
        print(model[0][:-4] +'-' + model[1] +',' + result.__str__())

    with open('ModelResults.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
           writer.writerow(key[0][:-4] +'-' + key[1] +',' + value.__str__())

def calculateHealtyAndOtherResult(fileName, encoderName, labelFilter):
    activeTrainingFilePath = os.path.join(root,fileName)
    data = pd.read_csv(activeTrainingFilePath, header=0, sep=',')
    y= data['Subtype']
    trainData,testData = train_test_split(data, test_size = 0.2, stratify=y)
    #testData = pd.read_csv(testDataAddress)
    dm = DataManager(trainData,testData,#numericColumnEncoderName=encoderName,numericColumns='All',
                     label='Subtype', labelFilter=labelFilter, encodeLabel=True)
    X, y = dm.GetTrainData()
    test_X, test_y = dm.GetTestData()
    le = dm.labelEncoder
    result = mmt.train(X, y, test_X, test_y, le)
    return  result

if __name__ ==  '__main__':
    #calculateAllModelResult()
    #basel = calculateHealtyAndOtherResult('RFE_50.csv', None, ['Basal', 'Normal'])
    #LumB = calculateHealtyAndOtherResult('RFE_50.csv',None,['LumB', 'Normal'] )
    #LumA = calculateHealtyAndOtherResult('RFE_50.csv',None,['LumA', 'Normal'] )
    #Her2 = calculateHealtyAndOtherResult('RFE_50.csv',None,['Her2', 'Normal'] )
    all = calculateHealtyAndOtherResult('RFE_50.csv', 'StandardScaler', ['Basal',  'LumB', 'LumA','Her2','Normal'])
    #print("basel:" + basel.__str__())
    #print("LumB:" + LumB.__str__())
    #print("LumA:" + LumA.__str__())
    #print("Her2:" + Her2.__str__())
    print("All:" + all.__str__())
