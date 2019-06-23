import lightgbm as lgb
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LgbmObjective(object):
    def __init__(self, X, y, metrics='mse'):
        self.X = X
        self.y = y
        self.metrics = metrics

    def __call__(self, trial):
        train_x, valid_x, train_y, valid_y = train_test_split(
            self.X, self.y, test_size=0.1)
        params = {
            'objective': 'regression',
            'metric': self.metrics,
            'boosting_type': 'gbdt',
            'n_estimators': 1000000,
            'lambda_l1': trial.suggest_uniform('lambda_l1', 0.0, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 0.0, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 8, 512),
            'bagging_fraction': trial.suggest_uniform(
                'bagging_fraction', 0.3, 1.0),
            'feature_fraction': trial.suggest_uniform(
                'feature_fraction', 0.3, 1.0),
            'learning_rate': trial.suggest_loguniform(
                'learning_rate', 1e-3, 0.1),
            'verbose': -1
        }

        lgbm = lgb.train(params, lgb.Dataset(train_x, train_y),
                         valid_sets=lgb.Dataset(valid_x, valid_y),
                         early_stopping_rounds=300,
                         verbose_eval=300)
        pred = lgbm.predict(valid_x)

        if self.metrics == 'rmse':
            return self._rmse(pred, valid_y)
        elif self.metrics == 'mae':
            return self._mae(pred, valid_y)
        else:
            return self._mape(pred, valid_y)

    def _rmse(self, y_pred, y_val, is_log=False):
        if is_log:
            return np.sqrt(mean_squared_error(
                np.expm1(y_pred), np.expm1(y_val)))
        else:
            return np.sqrt(mean_squared_error(y_pred, y_val))

    def _mae(self, y_pred, y_val, is_log=False):
        if is_log:
            return mean_absolute_error(np.expm1(y_pred), np.expm1(y_val))
        else:
            return mean_absolute_error(y_pred, y_val)

    def _mape(self, y_pred, y_val, is_log=False):
        if is_log:
            return np.mean(abs(
                np.expm1(y_pred)-np.expm1(y_val))/np.expm1(y_val))
        else:
            return np.mean(abs(y_val-y_pred)/y_val)


def optuna_tuning_lgbm(X, y, metrics='mae'):
    objective = LgbmObjective(X, y, metrics=metrics)

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=10)
    print(study.best_trial)
    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study
