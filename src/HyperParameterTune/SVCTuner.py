import optuna

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

Xtrain = None
ytrain = None

def objective(trial):
    svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    score = sklearn.model_selection.cross_val_score(classifier_obj, Xtrain, ytrain, n_jobs=-1, cv=5)
    accuracy = score.mean()
    return accuracy

def hyperParameterTune(XTrain, yTrain):
    global Xtrain
    global ytrain
    Xtrain = XTrain
    ytrain = yTrain
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=7)
    return study.best_trial.params['svc_c']


