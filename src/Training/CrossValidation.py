from numpy.core.einsumfunc import _einsum_path_dispatcher
import pandas as pd
from sklearn.model_selection import KFold
import os

trainDataAddress = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TrainDatas"
filelist = os.listdir(trainDataAddress)
for file in filelist:
    fileAddress = os.path.join(trainDataAddress , file)
    train_data = pd.read_csv(fileAddress)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_indicies, valid_indicies) in enumerate(kf.split(X = train_data)):
        train_data.loc[valid_indicies, "kfold"] = fold

    train_data.to_csv(fileAddress)