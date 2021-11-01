from lazypredict.Supervised import (  # pip install lazypredict
    LazyClassifier,
    LazyRegressor,
)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data and split
trainDataAddress = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\TrainDatas\FeatureSelection_20.csv"
train_data = pd.read_csv(trainDataAddress)
X = train_data.iloc[:,1:-1]
y = train_data.Subtype
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit LazyRegressor
reg = LazyClassifier(ignore_warnings=True, random_state=1121218, verbose=False)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)  # pass all sets

print(models.head(10))