
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

def GetEvalMetric(metricName, y_true, y_pred):
    if metricName == 'accuracy_score':
        return accuracy_score(y_true=y_true, y_pred=y_pred)
    elif metricName == 'precision_score':
        return precision_score(y_true=y_true, y_pred=y_pred) 
    elif metricName == 'recall_score':
        return recall_score(y_true=y_true, y_pred=y_pred)
    elif metricName == 'confusion_matrix':
            return confusion_matrix(y_true=y_true, y_pred=y_pred)
    else:
        raise Exception("Metric not found")
