import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\Selected_RFE_50"
trainPath = root + r"\RFE_50.csv"
train = pd.read_csv(trainPath,  header=0)
data = train.iloc[:,:-1]
sns.set(style='whitegrid')
corrMatrix = data.corr()
corrMatrix.to_csv(r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\Selected_RFE_50" + "/corrMatrix.csv", sep=";")
sns.heatmap(corrMatrix)
plt.show()