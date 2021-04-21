#!/usr/bin/env python
# coding: utf-8
# Loading Libraries and data
import sys
import unittest
from SvmModel import Model
from Hyperoptsvm import controller

# BaseClass


class TestSVMHyperopt(unittest.TestCase):
    """
    This class defines a different methods which are used to tests
    the various values.
    """

    def test_test_svm_degree(self):
        """
        Tests method which take input parameters .
        Tests the values of methods aganist known values.
         """
        train_path = "./data_transformed_10.csv"
        test_path = "./data_transformed_10.csv"
        obj = controller(Model(train_path, test_path, CV=5, label='Class'))
        expected_best_params = {
            'C': 1.77,
            'coef': 1.6216340381955197,
            'degree': 8.0,
            'kernel': 'poly',
            'gamma': 'scale'}
        self.assertEqual(
            expected_best_params['degree'],
            obj.optimize_hyperparam()['degree'])

    def test_test_svm_C(self):
        """
        Tests method which take input parameters .
        Tests the values of methods aganist known values.
         """
        train_path = "./data_transformed_10.csv"
        test_path = "./data_transformed_10.csv"
        obj = controller(Model(train_path, test_path, CV=5, label='Class'))
        expected_best_params = {
            'C': 1.7718619582441852,
            'coef0': 1.6216340381955197,
            'degree': 8.0,
            'kernel': 'poly',
            'gamma': 'scale'}
        self.assertEqual(
            expected_best_params['C'],
            obj.optimize_hyperparam()['C'])

    def test_test_svm_kernel(self):
        """
        Tests method which take input parameters .
        Tests the values of methods aganist known values.
         """
        train_path = "./data_transformed_10.csv"
        test_path = "./data_transformed_10.csv"
        obj = controller(Model(train_path, test_path, CV=5, label='Class'))
        expected_best_params = {
            'C': 1.7718619582441852,
            'coef': 1.6216340381955197,
            'degree': 8.0,
            'kernel': 'poly',
            'gamma': 'scale'}
        self.assertEqual(
            expected_best_params['kernel'],
            obj.optimize_hyperparam()['kernel'])

    def test_test_svm_coef(self):
        """
        Tests method which take input parameters .
        Tests the values of methods aganist known values.
         """
        train_path = "./data_transformed_10.csv"
        test_path = "./data_transformed_10.csv"
        obj = controller(Model(train_path, test_path, CV=5, label='Class'))
        expected_best_params = {
            'C': 1.7718619582441852,
            'coef0': 1.6216340381955197,
            'degree': 8.0,
            'kernel': 'poly',
            'gamma': 'scale'}
        self.assertEqual(
            expected_best_params['coef0'],
            obj.optimize_hyperparam()['coef0'])

    def test_test_svm_gamma(self):
        """
        Tests method which take input parameters .
        Tests the values of methods aganist known values.
         """
        train_path = "./data_transformed_10.csv"
        test_path = "./data_transformed_10.csv"
        obj = controller(Model(train_path, test_path, CV=5, label='Class'))
        expected_best_params = {
            'C': 1.7718619582441852,
            'coef': 1.6216340381955197,
            'degree': 8.0,
            'kernel': 'poly',
            'gamma': 'scale'}
        self.assertEqual( 
            expected_best_params['gamma'],
            obj.optimize_hyperparam()['gamma'])


# test suite is used to aggregate tests that should be executed together.
suite = unittest.TestLoader().loadTestsFromTestCase(TestSVMHyperopt)

# test runner which orchestrates the execution of tests and provide he outcome to the user.
# sys.stderr : Fileobject used by the interpreter for standard errors.
unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
