#!/usr/bin/env python
# coding: utf-8

"""
This script prompts a user to find best
hyperparameters values for Xgboost model by using hyperopt.
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt import space_eval
import xgboost as xgb


# baseclass


class XGBModel():
    """
      A class to represent XGboost and Hyperopt.

    """

    def __init__(
            self,
            train,
            test,
            num_boost_round=50,
            early_stopping_rounds=10,
            CV=5,
            label='Class'):
        """
        This method takes the input parameters, performs scaling
        """
        #train = pd.read_csv(train_path)  # reading train data
        #test = pd.read_csv(test_path)  # reading test data
        # remove :1000 during production
        self.X_train = train.iloc[:, :-1]
        self.X_test = test.iloc[:, :-1]
        self.y_train = train.iloc[:, :][label]
        self.y_test = test.iloc[:, :][label]        
        sc = StandardScaler()  # scaling data
        self.scaled_X_train = sc.fit_transform(self.X_train)
        self.scaled_X_test = sc.transform(self.X_test)
        self.dtrain = xgb.DMatrix(self.scaled_X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.scaled_X_test, label=self.y_test)
        label = self.dtrain.get_label()
        self.ratio = float(np.sum(label == 0)) / np.sum(label == 1)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.CV = CV

    # Define the function to minimize (XGboost Model)
    def optimize_hyperparam(self, n_eval=25):
        """
        This method is performing hyperparameter tuning using HyperOpt
        for XGboost
        ...
        parameters
        ----------
        In hyperopt bayesian optimization can be implemented giving three main
        parameters to the function fmin.
        Objective: Defines the loss function to minimize.
        space    : Defines the range of input values to test.
        Algo     : defines the search algorithm to select the best input values to
        use in each iteration.
        """
        def objective(space4xgb):
            """
            This method defines objective function to be tuned by the HyperOpt

            """
            model = xgb.train(
                space4xgb,
                self.dtrain,
                num_boost_round=self.num_boost_round,
                evals=[
                    (self.dtrain,
                     'train'),
                    (self.dtest,
                     'validation')],
                early_stopping_rounds=self.early_stopping_rounds)
            pred = model.predict(self.dtest)
            score = log_loss(self.y_test, pred)
            print("\tScore {0}\n\n".format(score))
            return {'loss': score, 'status': STATUS_OK}

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
                'booster': 'gbtree',
                'verbosity': 1,
                'disable_default_eval_metric': 1,
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

#model = XGBModel("data_transformed.csv", "data_transformed.csv")

#model.optimize_hyperparam()
