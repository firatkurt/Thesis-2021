
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def hyperParameterTune(X, y):
    best_k = 0
    best_score = 0
    neighbors = range(1,10,2)
    XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2)
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn. fit (XTrain, yTrain)
        y_pred = knn.predict(XValid)
        f1 = f1_score(yValid, y_pred, average='macro')
        if f1 > best_score:
            best_k = k
            best_score = f1
    return best_k