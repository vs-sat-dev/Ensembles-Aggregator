from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, mean_squared_error
import numpy as np
import optuna

from .CatBoostModel import CatBoostModel
from .LGBModel import LGBModel
from .TabnetModel import TabnetModel
from .XGBModel import XGBModel


class EnsemblesAggregator:
    """
    objective: binary, multiclass or regression
    """
    def __init__(self, x, y, objective_type, num_folds=5):
        self.x = x.copy()
        self.y = y.copy()
        self.num_folds = num_folds
        self.objective_type = objective_type

        self.models_dict = dict()
        self.preds_dict = dict()
        self.weights_dict = dict()
        self.models_id_dict = dict()
        self.names = []
        self.models = []

        if self.objective_type not in ['binary', 'multiclass', 'regression']:
            print('Wrong objective_type')
            exit()

        self.x.loc[:, 'fold'] = 0
        self.y.loc[:, 'fold'] = 0

        print(f'sxlen: {len(self.x)}')
        print(f'sylen: {len(self.y)}')

        kfolds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
        for fold, (_, val_ind) in enumerate(kfolds.split(self.x, self.y.drop('fold', axis=1))):
            self.x.loc[val_ind, 'fold'] = fold
            self.y.loc[val_ind, 'fold'] = fold

    def objective(self, trial):
        trial_arr = dict()
        all_weights = 0.0
        for name in self.names:
            trial_arr[f'{name}'] = (trial.suggest_float(f'{name}', 0.0, 1.0))
            all_weights += trial_arr[f'{name}']

        if all_weights < 1e-4:
            all_weights = 1e-4

        target_preds = np.zeros(len(self.x))
        for name in self.names:
            trial_arr[f'{name}'] = trial_arr[f'{name}'] / all_weights
            target_preds = target_preds + (self.preds_dict[f'{name}'] * trial_arr[f'{name}'])

        print(f'tuning_weights')

        if self.objective_type == 'multiclass':
            target_preds = np.argmax(target_preds, axis=1)
            return f1_score(self.y, target_preds, average='weighted')
        elif self.objective_type == 'binary':
            #print(f'tpreds: {target_preds}')
            #print(f'tpredshape: {target_preds.shape}')
            #target_preds = np.rint(target_preds)
            return roc_auc_score(self.y.drop('fold', axis=1), target_preds)
        elif self.objective_type == 'regression':
            return mean_squared_error(self.y, target_preds)

    def optuna_weights(self, num_trials=1000):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=num_trials)

        self.weights_dict = study.best_trial.params

        all_weights = 0.0
        for name in self.names:
            all_weights += self.weights_dict[name]

        for name in self.names:
            self.weights_dict[name] = self.weights_dict[name] / all_weights

    def fit(self, num_trials=100, models_types=['catboost', 'lightgbm', 'xgboost', 'tabnet']):
        """self.models = [
            #CatBoostModel(self.x, self.y, self.objective_type, self.num_folds),
            LGBModel(self.x, self.y, self.objective_type, self.num_folds),
            #XGBModel(self.x, self.y, self.objective_type, self.num_folds),
            TabnetModel(self.x, self.y, self.objective_type, self.num_folds)
        ]"""
        self.models = []
        for str_model in models_types:
            if str_model == 'catboost':
                self.models.append(CatBoostModel(self.x, self.y, self.objective_type, self.num_folds))
            elif str_model == 'lightgbm':
                self.models.append(LGBModel(self.x, self.y, self.objective_type, self.num_folds))
            elif str_model == 'xgboost':
                self.models.append(XGBModel(self.x, self.y, self.objective_type, self.num_folds))
            elif str_model == 'tabnet':
                self.models.append(TabnetModel(self.x, self.y, self.objective_type, self.num_folds))


        metric_funcs = None
        if self.objective_type == 'binary':
            metric_funcs = [
                f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, None
            ]
        elif self.objective_type == 'multiclass':
            metric_funcs = [
                f1_score, accuracy_score, precision_score, recall_score, None
            ]

        for e, model in enumerate(self.models):
            for metric_func in metric_funcs:
                func_name = 'none'
                if metric_func:
                    func_name = metric_func.__name__
                name = f'{model}_{func_name}'
                self.models_id_dict[f'{name}'] = e
                print(f'CurrentType: {name}')
                self.names.append(name)
                self.preds_dict[f'{name}'], self.models_dict[f'{name}'] = \
                    model.fit(num_trials=num_trials, metric_func=metric_func)

        self.optuna_weights(num_trials=num_trials*10)
        print(f'mdict: {self.models_dict}')

    def predict(self, x):
        out = np.zeros(len(x))
        for name in self.names:
            #print(f'name: {name}')
            #print(f'models: {self.models[self.models_id_dict[name]]}')
            #print(f'models_id: {self.models_id_dict[name]}')
            #print(f'models_dict: {self.models_dict[name]}')
            #print(f'weights_dict: {self.weights_dict[name]}')
            out = out + (self.models[self.models_id_dict[name]].predict(x, self.models_dict[name]) * self.weights_dict[name])
            #print(f'out: {out}')
        return out
