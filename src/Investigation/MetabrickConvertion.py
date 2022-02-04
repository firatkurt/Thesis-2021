import numpy as nm;
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\SourceData\METABRIC-Test.csv", sep=';')
print(df.describe())

df = df.T
result = df.iloc[:,1:]
result = pd.concat([result,df[0]], axis=1)