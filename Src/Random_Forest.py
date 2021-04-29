#!/usr/bin/env python
# coding: utf-8

"""
This script prompts a user to find best
hyperparameters values for Random Forest
model by using hyperopt.
"""


#import packages
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from hyperopt import space_eval
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

warnings.filterwarnings('ignore')


class RFModel():
    """
    A class to represent Random Forest and Hyperopt.
    """

    def __init__(self, train, test, CV=5, label='Class'):
        """
        Initializes Random Forest dataset object
        """
        #train = pd.read_csv(train_path)  # reading train data
        #test = pd.read_csv(test_path)  # reading test data
        self.X_train = train.iloc[:, :-1]
        self.X_test = test.iloc[:, :-1]
        self.y_train = train.iloc[:, :][label]
        self.y_test = test.iloc[:, :][label]
        sc = StandardScaler()  # scaling data
        self.scaled_X_train = sc.fit_transform(self.X_train)
        self.scaled_X_test = sc.transform(self.X_test)
        self.CV = CV

    def randomforest(self, hyper_parameter={}):
        """
        Applies cross_validation for the Random Forest algorithm
        """
        clf = RandomForestClassifier(**hyper_parameter,
                                     random_state=123)
        clf.fit(self.scaled_X_train, self.y_train)
        y_pred = clf.predict(self.scaled_X_test)
        scores = cross_val_score(
            clf,
            self.scaled_X_train,
            self.y_train,
            cv=self.CV,
            scoring='f1_macro')
        return np. mean(scores)

    # Define the function to minimize (Random Forest Model)
    def optimize_hyperparam(self, n_eval=10):
        """
        This method is performing hyperparameter tuning using HyperOpt
        for Random Forest
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

            space4rf = {
                'max_depth': hp.choice('max_depth', np.arange(5, 17, dtype=int)), 
                'n_estimators': hp.choice('n_estimators', np.arange(150, 200, dtype=int)), 
                'max_features': hp.choice('max_features', range(1, 10)), 
                'criterion': hp.choice('criterion', ["gini", "entropy"]), 
                'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(2, 12, dtype=int))
                }

            return space4rf
        space4rf = hyper_params()

        # defining function to optimize
        def objective(space4rf):
            """
            This method defines objective function to be tuned by the HyperOpt

            """
            f1 = rf_model.randomforest(space4rf)
            print(f1)
            return {'loss': -f1, 'status': STATUS_OK}

        # Initialize trials object.
        trials = Trials()
        # using seed to get repeatable results.
        seed = 123
        # run the hyper paramter tuning.
        best = fmin(fn=objective,
                    space=space4rf,
                    algo=tpe.suggest,
                    max_evals=n_eval,
                    trials=trials,
                    rstate=np.random.RandomState(seed))
        print(space_eval(space4rf, best))
        hyperparams = space_eval(space4rf, best)
        return hyperparams


# applying model on data file
#rf_model = RFModel("data_transformed.csv", "data_transformed.csv")


#rf_model.optimize_hyperparam()
