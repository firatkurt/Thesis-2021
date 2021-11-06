from xgboost import XGBClassifier
from sklearn.metrics import precision_score
import optuna

dm = None

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
        objective="multi:softmax",
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
    )
    xtrain,xvalid,ytrain,yvalid = dm.TrainTestSplit()
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    preds_valid = model.predict(xvalid)
    prec =  precision_score(yvalid,preds_valid, average='macro')
    return prec

def hyperParameterTune(dataModel):
    dm  = dataModel  
    study = optuna.create_study(direction='maximize')
    study.optimize(run, n_trials=50)
    return study.best_trial.params