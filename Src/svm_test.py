#!/usr/bin/env python
# coding: utf-8

"""
This script prompts a user to test SVM model.
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score


np.random.seed(123)

# import data
data_transformed = pd.read_csv('data_transformed.csv')

# avoid this ugly slicing by using a two-dim dataset
X = data_transformed.iloc[:5000, :-1]
y = data_transformed.iloc[:5000, :]['Class']


# using 75% of the data for training and 30% for testing (with
# stratification for imbalanced class)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=123)

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
scaled_X_train = sc.fit_transform(X_train)
scaled_X_test = sc.transform(X_test)

hyperparams = {
    'C': 1.7718619582441852,
    'coef0': 1.6216340381955197,
    'degree': 8.0,
    'gamma': 'scale',
    'kernel': 'poly'}


class Test:
    """
    A class to represent trains the model on best paramters and
    evaluate SVM model

    """

    def __init__(self, hyperparams, scaled_X_train, scaled_X_test, y_train):
        clf = svm.SVC(**hyperparams, class_weight='balanced', random_state=123)
        clf.fit(scaled_X_train, y_train)
        self.pred = clf.predict(scaled_X_test)

    def score(self, y_test):
        # Compute confusion matrix, roc_auc_score, Cohen_kappa_score

        scores = {"confusion_matrix": confusion_matrix(y_test, self.pred),
                  "roc_auc_score": roc_auc_score(y_test, self.pred),
                  "cohen_score": cohen_kappa_score(y_test, self.pred)}
        return scores


testing = Test(
    hyperparams=hyperparams,
    scaled_X_train=scaled_X_train,
    scaled_X_test=scaled_X_test,
    y_train=y_train)

print(testing.score(y_test=y_test))
