#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

np.random.seed(123)


# import data.
data_transformed = pd.read_csv('data_transformed.csv')

# avoid this ugly slicing by using a two-dim dataset.
X = data_transformed.iloc[:, :-1]
y = data_transformed.iloc[:, :]['Class']


# using 75% of the data for training and 25% for testing (with
# stratification for imbalanced class).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=123)

# Standardize features by removing the mean and scaling to unit variance.
sc = StandardScaler()
scaled_X_train = sc.fit_transform(X_train)
scaled_X_test = sc.transform(X_test)


class Hyperot:

    def __init__(self, best):
        self.best = best
        self.load_dataset()

        def load_dataset(self):
            # import data.
            data_transformed = pd.read_csv('data_transformed.csv')

            # avoid this ugly slicing by using a two-dim dataset.
            X = data_transformed.iloc[:, :-1]
            y = data_transformed.iloc[:, :]['Class']

            # using 75% of the data for training and 25% for testing (with
            # stratification for imbalanced class).
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=123)

    # Defining the space for hyperparameter tuning.

    '''
    hp.choice(label, options) — Returns one of the options, which should be a list or tuple.
    hp.loguniform(label, low, high) — Returns a value drawn according to exp(uniform(low, high))
    so that the logarithm of the return value is uniformly distributed.When optimizing,
    this variable is constrained to the interval [exp(low), exp(high)].

    '''

    space = {
        'C': hp.loguniform("C", np.log(1), np.log(100)),
        'kernel': hp.choice('kernel', ['rbf', 'poly']),
        'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
    }

    # SVM Parameters.

    '''
    C — Regularization parameter. The strength of the regularization is inversely proportional to C.
    kernel — Specifies the kernel type to be used in the algorithm.
    gamma — Kernel coefficient for ‘rbf’, ‘poly’.


    '''

    # Define the function to minimize (SVM Model).

    def objective(space):
        clf = svm.SVC(
            C=space['C'],
            kernel=space['kernel'],
            gamma=space['gamma'])

        # Fit the model on training set.
        clf.fit(scaled_X_train, y_train)
        # Make a prediction.
        pred = clf.predict(scaled_X_test)
        # Calculate our Metric - accuracy.
        accuracy = accuracy_score(y_test, pred)
        print("SCORE:", accuracy)

        # Because fmin() tries to minimize the objective, this function must
        # return the negative accuracy.
        return {'loss': -accuracy, 'status': STATUS_OK}

    # Initialize trials object.
    trials = Trials()
    # using seed to get repeatable results.
    seed = 123
    # run the hyper paramter tuning.
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=10,
                trials=trials,
                rstate=np.random.RandomState(seed))

    print(best)
