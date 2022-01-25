from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score
import optuna
import numpy as np
import pandas as pd


class CatBoostModel:
    def __init__(self, x, y, objective_type, num_folds, fold_feature='fold', params=None, models=None):
        self.x = x
        self.y = y
        self.num_folds = num_folds
        self.fold_feature = fold_feature
        self.params = params
        self.models = models
        self.objective_type = objective_type

    def __repr__(self):
        return CatBoostModel.__name__

    def train(self, params):
        if self.objective_type == 'regression':
            self.models = [CatBoostRegressor(**params) for _ in range(self.num_folds)]
        else:
            self.models = [CatBoostClassifier(**params) for _ in range(self.num_folds)]

        full_preds = np.zeros(len(self.y))

        for fold in range(self.num_folds):
            x_train = self.x.loc[self.x[self.fold_feature] != fold].drop(self.fold_feature, axis=1)
            y_train = self.y.loc[self.y[self.fold_feature] != fold].drop(self.fold_feature, axis=1)
            x_valid = self.x.loc[self.x[self.fold_feature] == fold].drop(self.fold_feature, axis=1)
            y_valid = self.y.loc[self.y[self.fold_feature] == fold].drop(self.fold_feature, axis=1)
            self.models[fold].fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0, early_stopping_rounds=100)

            if self.objective_type == 'regression':
                full_preds[y_valid.index] = self.models[fold].predict(x_valid)
            else:
                full_preds[y_valid.index] = self.models[fold].predict_proba(x_valid)[:, 1]

        return full_preds

    def objective(self, trial, metric_func):

        param = {
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        if self.objective_type == 'binary':
            param["objective"] = "Logloss"
            preds = self.train(param)
            return metric_func(self.y.drop(self.fold_feature, axis=1), np.rint(preds))
        elif self.objective_type == 'regression':
            param["objective"] = "RMSE"
            preds = self.train(param)

            try:
                res = metric_func(self.y.drop(self.fold_feature, axis=1), preds)
            except:
                res = metric_func(self.y.drop(self.fold_feature, axis=1), np.abs(preds))

            return res
        else:
            print('Wrong objective_type CatBoost')
            exit()

    def fit(self, metric_func, direction_func, num_trials=100):
        if metric_func:
            objective_caller = lambda trials: self.objective(trials, metric_func)
            study = optuna.create_study(direction=direction_func)
            study.optimize(objective_caller, n_trials=num_trials)
            self.params = study.best_trial.params
        else:
            if self.objective_type == 'binary' or self.objective_type == 'multiclass':
                self.models = [CatBoostClassifier() for _ in range(self.num_folds)]
            elif self.objective_type == 'regression':
                self.models = [CatBoostRegressor() for _ in range(self.num_folds)]
            self.params = dict()

        preds = self.train(self.params)
        #print(f'CatBoostScore: {accuracy_score(self.y.drop(self.fold_feature, axis=1), np.rint(preds))}')

        #df = pd.DataFrame(preds)
        #df.columns = ['catboost']

        return preds, self.models

    def predict(self, x, models=None):
        if models:
            self.models = models
        preds = np.zeros(len(x))
        for i in range(self.num_folds):
            if self.objective_type == 'regression':
                preds += self.models[i].predict(x)
            else:
                preds += self.models[i].predict_proba(x)[:, 1]
        return preds / self.num_folds
    """preds = pd.DataFrame()
    preds.index = range(len(x))
    preds['catboost'] = 0.0
    for i in range(self.num_folds):
        preds['catboost'] = preds['catboost'] + pd.Series(self.models[i].predict_proba(x)[:, 1])

    preds['catboost'] = preds['catboost'] / float(self.num_folds)

    return preds"""

