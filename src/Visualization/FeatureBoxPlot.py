import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

trainPath = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\Selected_RFE_50\RFE_50.csv"
train = pd.read_csv(trainPath,  header=0)
columns = ['FOXA1', 'CEP55', 'KRT5', 'PTTG1','COL17A1', 'ESR1', 'CDK1', 'ERBB2', 'GATA3', 'TSHZ2', 'RERG', 'AGR3', 'KRT14','SLC39A6','SFRP1']
#all = ['FOXA1', 'CEP55', 'KRT5', 'PTTG1','COL17A1', 'ESR1', 'CDK1', 'ERBB2', 'GATA3', 'TSHZ2', 'RERG', 'AGR3', 'KRT14', 'Subtype' ]
y = ['Subtype']
all = columns + y
data = train[all]
figure, axis = plt.subplots(5, 3)
sns.set(style='whitegrid')
#sns.boxplot(x=columns[0], y="Subtype", data=data, ax = axis[0][0])
#sns.boxplot(x=columns[1], y="Subtype", data=data, ax = axis[0][1])

for i in range(len(columns)):
    y = int(i%3)
    x = int(i/3)
    sns.boxplot(x="Subtype", y=columns[i], data=data, ax =axis[x][y])
#data.boxplot(by ='Subtype', column =['FOXA1'], grid = False)
plt.show()