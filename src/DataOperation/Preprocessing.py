import numpy as nm
import pandas as pd
from sklearn.model_selection import train_test_split

#df = pd.read_csv("C:\\Users\\FIRAT.KURT\\Documents\\Thesis_2021\\TestData.csv")

df = pd.read_csv(r"/Users/firatkurt/Documents/Thesis_Data/SourceData/BRCA_exp_subtype_T.csv")
print(len(df.values))
print(len(df.columns))
#df = df.T.to_csv("BRCA_exp_subtype_T.csv",index=True, header=False)
print(df.columns)

result = df.iloc[:,2:]
df.Subtype[1:114] = 'Healty'
print(df.groupby("Subtype").size())
result = pd.concat([result,df.Subtype], axis=1)

result = result.dropna(axis=0)
print(len(result.values))
print(len(result.columns))

train_Data, test_Data= train_test_split(result, train_size=900, random_state = 0)
test_Data = pd.DataFrame(test_Data)
train_Data = pd.DataFrame(train_Data)
print(len(train_Data.values))
print(train_Data.groupby("Subtype").size())
print(len(test_Data.values))
print(test_Data.groupby("Subtype").size())

train_Data.reset_index(drop=True)
train_Data.to_csv("TrainData.csv",index=True, header=True)

test_Data.reset_index(drop=True)
  
test_Data.to_csv("TestData.csv",index=True, header=True)