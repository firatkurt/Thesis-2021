import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

# Import some data to play with
#root = r"/Users/firatkurt/Documents/Thesis_Data/Selected_RFE_50"
#trainPath = root + r"/RFE_50.csv"
root = r"/Users/firatkurt/Documents/Thesis_Data/RFE50_BRCA"
trainPath = root + r"/RFE50_BRCA_train.csv"

data = pd.read_csv(trainPath, header=0)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

le = LabelEncoder()
y = le.fit_transform(y)
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2, 3, 4])
n_classes = y.shape[1]


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(
    svm.SVC(kernel="linear", probability=True, random_state=42)
)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0],tpr[0],color="b",lw=lw,label="%s ROC curve (area = %0.2f)" % (le.inverse_transform([0])[0], roc_auc[0]),)
plt.plot(fpr[1],tpr[1],color="g",lw=lw,label="%s ROC curve (area = %0.2f)" % (le.inverse_transform([1])[0],roc_auc[1]),)
plt.plot(fpr[2],tpr[2],color="c",lw=lw,label="%s ROC curve (area = %0.2f)" % (le.inverse_transform([2])[0],roc_auc[2]),)
plt.plot(fpr[3],tpr[3],color="r",lw=lw,label="%s ROC curve (area = %0.2f)" % (le.inverse_transform([3])[0],roc_auc[3]),)
plt.plot(fpr[4],tpr[4],color="y",lw=lw,label="%s ROC curve (area = %0.2f)" % (le.inverse_transform([4])[0],roc_auc[4]),)
plt.plot(fpr["micro"],tpr["micro"],color="darkorange",lw=lw,label="ROC curve (area = %0.2f)" % roc_auc["micro"],)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()