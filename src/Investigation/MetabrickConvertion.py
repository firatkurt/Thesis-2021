import numpy as nm;
import pandas as pd
from sklearn.model_selection import train_test_split

#df = pd.read_csv(r"/Users/firatkurt/Documents/Thesis_Data/MetaBrickData/RFE_50.csv")
#print(df.describe())

#df = df.T
#result = df.iloc[:,1:]
#result = pd.concat([result,df[0]], axis=1)
columns = ['NAT1', 'BCL2', 'BUB1', 'CA12', 'CCNA2', 'CDK1', 'CENPE', 'COL17A1',
       'EGFR', 'ERBB2', 'ESR1', 'FGFR4', 'FOXC1', 'GABRP', 'GATA3', 'FOXA1',
       'KRT5', 'KRT14', 'KRT17', 'PRNP', 'CX3CL1', 'SFRP1', 'SIAH2', 'SOX10',
       'XBP1', 'CCNB2', 'PTTG1', 'KIF23', 'MELK', 'UBE2C', 'TBC1D9', 'SLC39A6',
       'SPDEF', 'KLK5', 'PTTG3P', 'BCL11A', 'ANLN', 'ROPN1', 'CEP55', 'MCM10',
       'RGMA', 'MLPH', 'RERG', 'REEP6', 'CDCA5', 'TSHZ2', 'OSR1',
       'CKAP2L', 'AGR3', 'Subtype']
df_train = pd.read_csv(r"/Users/firatkurt/Documents/Thesis_Data/SourceData/TrainData.csv")
df_test = pd.read_csv(r"/Users/firatkurt/Documents/Thesis_Data/SourceData/TestData.csv")

df_filtered_train = df_train[columns]
df_filtered_test = df_test[columns]

dest_url = r"/Users/firatkurt/Documents/Thesis_Data/RFE50_BRCA"

df_filtered_train.to_csv(dest_url + '/RFE50_BRCA_train.csv', index=False, header=True)
df_filtered_test.to_csv(dest_url + '/RFE50_BRCA_test.csv', index=False, header=True)