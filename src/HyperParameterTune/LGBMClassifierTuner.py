from lightgbm import LGBMClassifier as lgbm
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import optuna

XTrain = None
XValid = None
yTrain = None
yValid = None
average = 'binary'

def run(trial):
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }

    model = lgbm(**param_grid)
    model.fit(XTrain,yTrain)
    preds_valid = model.predict(XValid)
    prec = precision_score(yValid, preds_valid, average=average)
    return prec

def hyperParameterTune(X, y):
    global XTrain
    global XValid
    global yTrain
    global yValid
    global  average
    XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2)
    average = 'binary'
    if len(yValid.iloc[:,0].unique()) > 2:
        average = 'macro'
    study = optuna.create_study(direction='maximize')
    study.optimize(run, n_trials=7)
    return study.best_trial.params

