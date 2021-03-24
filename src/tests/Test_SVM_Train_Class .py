#!/usr/bin/env python
# coding: utf-8
# Import statements
import sys
import unittest
from svm_model import Model

# BaseClass


class TestSVM(unittest.TestCase):
    """
    This class defines a different methods which are used to tests
    the various values.
    """
# Test methods

    def test_train_svm_linear(self):
        """
        Tests method which take input parameters .
        Tests the values of methods aganist known values.
         """
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
        """
        Tests method which take input parameters .
        Tests the values of methods aganist known values.
         """
        train_path = "./data_transformed_10.csv"
        test_path = "./data_transformed_10.csv"
        obj = Model(train_path, test_path, CV=5, label='Class')
        result2 = obj.svm(
            hyper_parameter={
                'kernel': 'poly',
                'C': 1.0,
                'gamma': 'scale'})
        expected_cv_result2 = 0.60
        self.assertEqual(expected_cv_result2, round(result2, 2))

    def test_train_svm_rbf(self):
        """
        Tests method which take input parameters .
        Tests the values of methods aganist known values.
         """
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

#test suite is used to aggregate tests that should be executed together.
suite = unittest.TestLoader().loadTestsFromTestCase(TestSVM)
#test runner which orchestrates the execution of tests and provide he outcome to the user.
unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
