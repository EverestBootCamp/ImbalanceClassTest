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
from SVM_trainclass.T103 import Model
from T102_Annotations.data_annotation import Annotation



def arguments():
    '''get input arguments'''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("--algorithm", default="svm", help="algorithm for modelling", type=str, choices=['svm'])
    parser.add_argument("--percentage", default=0.1, help="percentage of data used for training", type=float)



    return parser.parse_args()


def main():

    if(args.algorithm == 'svm'):

        logging.basicConfig(filename='logfile.log', filemode="w", format ='%(asctime)s %(message)s' ,level=logging.INFO)
        percentage = args.percentage

        logging.info('*******Creating test and train data sets*******')
        train_df_Class = Annotation(test_size=percentage,output_dataframe=str(percentage))
        train_df = train_df_Class.dataframe()
        test_df_Class = Annotation(test_size=percentage,output_dataframe='test')
        test_df = test_df_Class.dataframe()
        logging.info('*******Creating the SVM model*******')
        svm_model = Model(train=train_df,test=test_df)
        logging.info('*******Optimizing the Hyper Parameters*******')
        hyperOpt_class = controller(svm_model)
        hyperOpt_class.optimize_hyperparam()
    



if __name__ == '__main__':
    args=arguments()
    main()
    

