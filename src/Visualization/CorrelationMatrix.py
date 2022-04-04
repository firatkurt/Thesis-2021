import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root = r"/Users/firatkurt/Documents/Thesis_Data/Selected_RFE_50"
trainPath = root + r"/RFE_50.csv"
train = pd.read_csv(trainPath,  header=0)
data = train.iloc[:,:-1]
#sns.set_theme()
#sns.palplot(sns.diverging_palette(240, 10, n=9))
corrMatrix = data.corr()
corrMatrix.to_csv(root + "/corrMatrix.csv", sep=";")
#sns.heatmap(corrMatrix)
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), cmap="RdBu", xticklabels = 1, yticklabels= 1, ax=ax)
plt.show()