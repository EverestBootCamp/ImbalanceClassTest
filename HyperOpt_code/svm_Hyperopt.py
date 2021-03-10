#!/usr/bin/env python
# coding: utf-8
"""
This script prompts a user to find best
hyperparameters values for SVM model by using hyperopt.
"""


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt import space_eval


class Model():
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

    def __init__(self, train_path, test_path, label='Class'):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        # remove :1000 during production
        self.X_train = train.iloc[:1000, :-1]
        self.X_test = test.iloc[:1000, :-1]
        self.y_train = train.iloc[:1000, :][label]
        self.y_test = test.iloc[:1000, :][label]
        # self.data = self.data.rename(columns={"Unnamed: 0": "T0"})

        # defining function for scaling data
    def scaler(self):
        sc = StandardScaler()
        self.scaled_X_train = sc.fit_transform(self.X_train)
        self.scaled_X_test = sc.transform(self.X_test)

        # definig the SVM model and corresponding parameters
    def svm(self, hyper_parameter={}):

        clf = svm.SVC(
            **hyper_parameter,
            class_weight='balanced',
            random_state=0)

        clf.fit(self.scaled_X_train, self.y_train)
        self.y_pred = clf.predict(self.scaled_X_test)
        return f1_score(self.y_test, self.y_pred)


# Defining the space for hyperparameter tuning.
space4svm = {
    'C': hp.uniform('C', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
    'gamma': hp.choice('gamma', ['scale', 'auto']),
    'degree': hp.quniform('degree', 3, 10, 1),
    'coef0': hp.uniform('coef0', 0, 10)
}


def objective(space4svm):
    """
    hp.choice(label, options) — Returns one of the options, which should be a list or tuple.
    hp.randint(label, upper) — Returns a random integer between the range [0, upper).
    hp.uniform(label, low, high) — Returns a value uniformly between low and high.
    hp.quniform(label, low, high, q) — Returns a value round(uniform(low, high) / q) * q,
    i.e it rounds the decimal values and returns an integer.
    hp.normal(label, mean, std) — Returns a real value that’s normally-distributed with
    mean and standard deviation sigma.
    """
    model = Model(
        "data_transformed_annotated_20pct.csv",
        "data_transformed_annotated_10pct.csv")
    # "data_transformed.csv",
    # "data_transformed.csv")
    model.scaler()
    space4svm['degree'] = int(space4svm['degree'])
    print(space4svm)
    f1 = model.svm(space4svm)
    print(f1)
    return {'loss': -f1, 'status': STATUS_OK}


# Initialize trials object.
trials = Trials()
# using seed to get repeatable results.
SEED = 123
# run the hyper paramter tuning.
best = fmin(fn=objective,
            space=space4svm,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials,
            rstate=np.random.RandomState(SEED))

print(best)
print(space_eval(space4svm, best))
