#!/usr/bin/env python
# coding: utf-8

"""
This script prompts a user to find best
hyperparameters values for SVM model by using hyperopt.
"""

import warnings
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt import space_eval
from sklearn.metrics import f1_score
from sklearn import svm
import pandas as pd
import numpy as np
#from SvmModel import Model

#model = Model("data_transformed.csv",
#              "data_transformed.csv")


warnings.filterwarnings('ignore')


class controller():
    """
    A class to represent hyperot.
    ...
    Attributes
    ----------
    In hyperopt bayesian optimization can be implemented giving three main
    parameters to the function fmin.
    Objective: Defines the loss function to minimize.
    space    : Defines the range of input values to test.
    Algo     : defines the search algorithm to select the best input values to
    use in each iteration.
    """

    def __init__(self, model):
        self.model = model

    # Define the function to minimize (SVM Model)
    def optimize_hyperparam(self, n_eval=10):


        def objective(space4svm):
            f1 = self.model.svm(space4svm)
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
            space4svm = {
                'C': hp.uniform(
                    'C', 0, 20), 'kernel': hp.choice(
                    'kernel', [
                        'linear', 'sigmoid', 'poly', 'rbf']), 'gamma': hp.choice(
                    'gamma', [
                        'scale', 'auto']), 'degree': hp.quniform(
                            'degree', 3, 10, 1), 'coef0': hp.uniform(
                                'coef0', 0, 10)}

            return space4svm
        space4svm = hyper_params()

        # Initialize trials object.
        trials = Trials()
        # using seed to get repeatable results.
        seed = 123
        # run the hyper paramter tuning.
        best = fmin(fn=objective,
                    space=space4svm,
                    algo=tpe.suggest,
                    max_evals=n_eval,
                    trials=trials,
                    rstate=np.random.RandomState(seed))
        print(space_eval(space4svm, best))
        hyperparams = space_eval(space4svm, best)
        return hyperparams


#hyperCl = controller(model)

#hyperCl.optimize_hyperparam()
