from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import optuna

XTrain = None
XValid = None
yTrain = None
yValid = None
average = 'binary'

def run(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)
    model = XGBClassifier(
        random_state=42,
        n_estimators=7000,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
    )

    model.fit(XTrain, yTrain, early_stopping_rounds=300, eval_set=[(XValid, yValid)], verbose=False)
    preds_valid = model.predict(XValid)
    prec =  precision_score(yValid,preds_valid, average=average)
    return prec


def hyperParameterTune(X, y):
    global XTrain
    global XValid
    global yTrain
    global yValid
    global average
    XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2)
    if len(yValid.iloc[:, 0].unique()) > 2:
        average = 'macro'
    study = optuna.create_study(direction='maximize')
    study.optimize(run, n_trials=7)
    return study.best_trial.params
