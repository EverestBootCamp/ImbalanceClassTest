#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import unittest
from svm_model import Model


# In[7]:


class TestSVM(unittest.TestCase):

    def test_train_svm_linear(self):
        train_path = "./data_transformed_10.csv"
        test_path = "./data_transformed_10.csv"
        obj = Model(train_path, test_path, CV=5, label='Class')
        result1 = obj.svm(
            hyper_parameter={
                'kernel': 'linear',
                'C': 1.0,
                'gamma': 'auto'})
        expected_cv_result1 = 0.70
        self.assertEqual(expected_cv_result1, round(result1, 2))

    def test_train_svm_poly(self):
        train_path = "./data_transformed_10.csv"
        test_path = "./data_transformed_10.csv"
        obj = Model(train_path, test_path, CV=5, label='Class')
        result2 = obj.svm(
            hyper_parameter={
                'kernel': 'poly',
                'C': 1.0,
                'gamma': 'scale'})
        expected_cv_result2 = 0.70
        self.assertEqual(expected_cv_result2, round(result2, 2))

    def test_train_svm_rbf(self):
        train_path = "./data_transformed_10.csv"
        test_path = "./data_transformed_10.csv"
        obj = Model(train_path, test_path, CV=5, label='Class')
        result = obj.svm(
            hyper_parameter={
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 0.7})
        expected_cv_result = 0.70
        self.assertEqual(expected_cv_result, round(result, 2))


# In[8]:

suite = unittest.TestLoader().loadTestsFromTestCase(TestSVM)
unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)


# In[ ]:
