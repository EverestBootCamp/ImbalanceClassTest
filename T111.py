import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import optuna
import optuna.integration.lightgbm as lgb
import math
def objective(trial):
    train = pd.read_csv("data_transformed.csv") #reading train data
    test = pd.read_csv("data_transformed.csv") #reading test data
    train_x= train.iloc[:1000,:-1]
    test_x= test.iloc[:1000,:-1]
    train_y = train.iloc[:1000,:]['Class'] 
    test_y = test.iloc[:1000,:]['Class']
    sc = StandardScaler() #scaling data
    scaled_X_train = sc.fit_transform(train_x)
    scaled_X_test = sc.transform(test_x)
    dtrain = lgb.Dataset(scaled_X_train, label = train_y, free_raw_data = False)
    dtest  = lgb.Dataset(scaled_X_test, label = test_y, free_raw_data = False)
    
    param = {
        'objective': 'poisson',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'force_row_wise': True,
        'max_depth': -1,
        
        'max_bin': trial.suggest_int('max_bin', 1, 512),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        
        'sub_feature': trial.suggest_uniform('sub_feature', 0.0, 1.0),
        'sub_row': trial.suggest_uniform('sub_row', 0.0, 1.0)
    }
    
    # Add a callback for pruning
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'rmse')
    
    gbm = lgb.train(
        param, 
        dtrain, 
        verbose_eval = 20,
        valid_sets = [dtest], 
        callbacks = [pruning_callback]
        )
    
    preds = gbm.predict(scaled_X_test)
    accuracy = (sklearn.metrics.mean_squared_error(test_y, preds))

    return accuracy
study = optuna.create_study(direction = 'minimize', pruner = optuna.pruners.MedianPruner(n_warmup_steps = 10))
study.optimize(objective, n_trials = 200)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))