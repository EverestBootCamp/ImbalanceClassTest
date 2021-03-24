#!/usr/bin/env python
# coding: utf-8

"""
This script prompts a user to find best
hyperparameters values for Xgboost model by using hyperopt.
"""


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt import space_eval
from sklearn.model_selection import cross_val_score
import xgboost
from xgboost import XGBClassifier


# baseclass


class XGModel():
    """
      A class to represent XGboost and Hyperopt.

    """

    def __init__(
            self,
            train_path,
            test_path,
            early_stopping_rounds=10,
            CV=5,
            label='Class'):
        """
        This method takes the input parameters, performs scaling
        """
        train = pd.read_csv(train_path)  # reading train data
        test = pd.read_csv(test_path)  # reading test data
        # remove :1000 during production
        self.X_train = train.iloc[:1000, :-1]
        self.X_test = test.iloc[:1000, :-1]
        self.y_train = train.iloc[:1000, :][label]
        self.y_test = test.iloc[:1000, :][label]
        sc = StandardScaler()  # scaling data
        self.scaled_X_train = sc.fit_transform(self.X_train)
        self.scaled_X_test = sc.transform(self.X_test)
        self.ratio = float(np.sum(label == 0)) / np.sum(label == 1)
        self.early_stopping_rounds = early_stopping_rounds
        self.CV = CV

    def xgb(self, hyper_parameter={}):
        """
        This method is fitting data and performing cross validation
        """

        clf = xgboost.XGBClassifier(**hyper_parameter)
        clf.fit(self.scaled_X_train, self.y_train)
        self.y_pred = clf.predict(self.scaled_X_test)
        scores = cross_val_score(
            clf,
            self.scaled_X_train,
            self.y_train,
            cv=self.CV,
            scoring='f1_macro')
        return np. mean(scores)

    # Define the function to minimize (XGboost Model)
    def optimize_hyperparam(self, n_eval=10):
        """
        This method is performing hyperparameter tuning using HyperOpt
        for XGboost
        """
        def objective(space4xgb):
            """
            This method performs f1 score

            """

            print(space4xgb)
            f1 = model.xgb(space4xgb)
            print(f1)
            return {'loss': -f1, 'status': STATUS_OK}
            f1 = model.xgb(space4xgb)
            print(f1)
            return {'loss': -f1, 'status': STATUS_OK}

        # Defining the space for hyperparameter tuning.
        def hyper_params():
            """
            hp.choice(label, options) — Returns one of the options, which should be a list or tuple.
            hp.randint(label, upper) — Returns a random integer between the range [0, upper).
            hp.uniform(label, low, high) — Returns a value uniformly between low and high.
            hp.quniform(label, low, high, q) — Returns a value round(uniform(low, high) / q) * q,
            i.e it rounds the decimal values and returns an integer.
            hp.normal(label, mean, std) — Returns a real value that’s normally-distributed with
            mean and standard deviation sigma.
            """
            space4xgb = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'verbosity': 1,
                'disable_default_eval_metric': 1,
                'booster': 'gbtree',
                'reg_lambda': hp.quniform('reg_lambda', 1, 2, 0.1),
                'reg_alpha': hp.quniform('reg_alpha', 0, 10, 1),
                'max_delta_step': hp.quniform('max_delta_step', 1, 10, 1),
                'max_depth': hp.choice('max_depth', np.arange(1, 14,
                                                              dtype=int)),
                'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
                'gamma': hp.quniform('gamma', 0.5, 1.0, 0.05),
                'sampling_method': 'uniform',
                'min_child_weight': hp.quniform('min_child_weight',
                                                1, 10, 1),
                'colsample_bytree': hp.quniform('colsample_bytree',
                                                0.5, 1, 0.05),
                'colsample_bylevel': hp.quniform('colsample_bylevel',
                                                 0.5, 1, 0.05),
                'colsample_bynode': hp.quniform('colsample_bynode',
                                                0.5, 1, 0.05),
                'scale_pos_weight': self.ratio

            }

            return space4xgb
        space4xgb = hyper_params()

        # Initialize trials object.
        trials = Trials()
        # using seed to get repeatable results.
        seed = 123
        # run the hyper paramter tuning.
        best = fmin(fn=objective,
                    space=space4xgb,
                    algo=tpe.suggest,
                    max_evals=n_eval,
                    trials=trials,
                    rstate=np.random.RandomState(seed))
        print(space_eval(space4xgb, best))
        hyperparams = space_eval(space4xgb, best)
        return hyperparams


model = XGModel("data_transformed.csv", "data_transformed.csv")

model.optimize_hyperparam()
