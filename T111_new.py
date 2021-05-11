#!/usr/bin/env python
# coding: utf-8

"""
This script prompts a user to find best
hyperparameter using optuna for LGBM.
"""

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import optuna
import optuna.integration.lightgbm as lgb
import math


def objective(trial):


    train = pd.read_csv("data_transformed.csv")  # reading train data
    test = pd.read_csv("data_transformed.csv")  # reading test data
    train_x = train.iloc[:1000, :-1]
    test_x = test.iloc[:1000, :-1]
    train_y = train.iloc[:1000, :]["Class"]
    test_y = test.iloc[:1000, :]["Class"]
    sc = StandardScaler()  # scaling data
    scaled_X_train = sc.fit_transform(train_x)
    scaled_X_test = sc.transform(test_x)
    dtrain = lgb.Dataset(scaled_X_train, label=train_y, free_raw_data=False)
    dtest = lgb.Dataset(scaled_X_test, label=test_y, free_raw_data=False)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 1, 512),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    pruning_callback = optuna.integration.LightGBMPruningCallback(
        trial, "binary_logloss"
    )

    gbm = lgb.train(
        param, dtrain, verbose_eval=20, valid_sets=[dtest], callbacks=[pruning_callback]
    )

    preds = gbm.predict(scaled_X_test)
    f1_score = sklearn.metrics.f1_score(test_y, preds)
    return f1_score

    if __name__ == "__main__":
        dtrain = lgb.Dataset(scaled_X_train, label=train_y, free_raw_data=False)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }

        tuner = lgb.LightGBMTunerCV(
            params,
            dtrain,
            verbose_eval=100,
            early_stopping_rounds=100,
            folds=KFold(n_splits=3),
        )

        tuner.run()

        print("Best score:", tuner.best_score)
        best_params = tuner.best_params
        print("Best params:", best_params)
        print("  Params: ")
        for key, value in best_params.items():
            print("    {}: {}".format(key, value))
