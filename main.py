#main file of the imbalancedClass project

#importing packages

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt import space_eval
from HyperOpt_code.Hyperoptsvm import controller
from XGBoost.XGboostnew import XGBModel
from SVM_trainclass.T103 import Model
from T102_Annotations.data_annotation import Annotation
from sklearn.metrics import log_loss
import xgboost as xgb
from LGBM.T111_modified import LGBM
import optuna
#import optuna.integration.lightgbm as lgb
import math
from optuna.trial import Trial
from Random_Forest.Random_Forest import RFModel
from sklearn.ensemble import RandomForestClassifier


def arguments():
    '''get input arguments'''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("--algorithm", default="xgb", help="algorithm for modelling", type=str, choices=['svm','xgb','lgbm','random_forest'])
    parser.add_argument("--percentage", default=0.1, help="percentage of data used for training", type=float)



    return parser.parse_args()


def main():
    logging.basicConfig(filename='logfile.log', filemode="w", format ='%(asctime)s %(message)s' ,level=logging.INFO)
    percentage = args.percentage

    logging.info('*******Creating test and train data sets*******')
    train_df_Class = Annotation(test_size=percentage,output_dataframe=str(percentage))
    train_df = train_df_Class.dataframe()
    test_df_Class = Annotation(test_size=percentage,output_dataframe='test')
    test_df = test_df_Class.dataframe()

    if (args.algorithm == 'svm'):

        
        logging.info('*******Creating the SVM model*******')
        svm_model = Model(train=train_df,test=test_df)
        logging.info('*******Optimizing the Hyper Parameters*******')
        hyperOpt_class = controller(svm_model)
        hyperOpt_class.optimize_hyperparam()
        logging.info('*******HyperOptimization Finished*******')

    elif (args.algorithm == 'xgb'):
        logging.info('*******Initializing the XGB model*******')
        xgb_model = XGBModel(train=train_df,test=test_df)
        logging.info('*******Optimizing the Hyper Parameters*******')
        xgb_model.optimize_hyperparam()
        logging.info('*******HyperOptimization Finished*******')
    
    elif (args.algorithm == 'lgbm'):
        logging.info('*******Initializing the LGBM model*******')
        X_train = train_df.iloc[:, :-1]
        X_test = test_df.iloc[:, :-1]
        y_train = train_df.iloc[:, :]['Class']
        y_test = test_df.iloc[:, :]['Class']
        sc = StandardScaler()
        scaled_X_train = sc.fit_transform(X_train)
        scaled_X_test = sc.transform(X_test)
        lgbm_class = LGBM (scaled_X_train,scaled_X_test,y_train,y_test)
        logging.info('*******Optimizing the Hyper Parameters*******')
        lgbm_class.optuna_method()
        logging.info('*******HyperOptimization Finished*******')

    elif (args.algorithm == 'random_forest'):
        logging.info('*******Initializing the Random Forest model*******')
        rf_model = RFModel(train_df,test_df)
        logging.info('*******Optimizing the Hyper Parameters*******')
        rf_model.optimize_hyperparam()
        logging.info('*******HyperOptimization Finished*******')














    



if __name__ == '__main__':
    args=arguments()
    main()
    

