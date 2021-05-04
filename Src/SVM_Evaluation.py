#!/usr/bin/env python
# coding: utf-8

"""
This script prompts a user to test SVM model.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Hyperoptsvm import controller
from SvmModel import Model


class Test:
    """
    A class to represent trains the model on best paramters and
    evaluate SVM model

    """

    def __init__(self, hyperparams, scaled_X_train, scaled_X_test, y_train):
        self.clf = svm.SVC(
            **hyperparams,
            class_weight='balanced',
            random_state=123,
            probability=True)
        self.clf.fit(scaled_X_train, y_train)
        self.pred = self.clf.predict(scaled_X_test)

    def score(self, y_test):
        # Compute confusion matrix, roc_auc_score, Cohen_kappa_score

        performance = {"confusion_matrix": confusion_matrix(y_test, self.pred),
                       "roc_auc_score": roc_auc_score(y_test, self.pred),
                       "cohen_score": cohen_kappa_score(y_test, self.pred)}
        return performance

    def roc(self):
        # ROC Curve
        clf = self.clf
        svc_disp = plot_roc_curve(clf, scaled_X_test, y_test)
        svc_disp.ax_.set_xlabel('False Positive Rate')
        svc_disp.ax_.set_ylabel('True Positive Rate')
        svc_disp.ax_.set_title('ROC curve')
        plt.show()
        plt.savefig('roc.png')

    def prcurve(self):
        #Precision Recall Curve
        clf = self.clf
        y_score = clf.decision_function(scaled_X_test)
        average_precision = average_precision_score(y_test, y_score)
        disp = plot_precision_recall_curve(clf, scaled_X_test, y_test)
        disp.ax_.set_xlabel('Recall')
        disp.ax_.set_ylabel('Precision')
        disp.ax_.set_title('2-class Precision-Recall curve: '
                           'AP={0:0.2f}'.format(average_precision))
        plt.show()
        plt.savefig('prcurve.png')
