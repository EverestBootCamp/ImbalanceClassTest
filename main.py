#main file of the imbalancedClass project

#importing packages

#import argeparse
#import logging
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


#Setting up log file and messages we want into the log file
#logging.basicConfig(filename='Log_File.log', encoding='utf-8', level=logging.DEBUG)


#Setting up arguments for the file and writing help for the main file 
#parser = argparse.ArgumentParser()
#parser.add_argument("classification_model", help="specify the algorithm for classification",
 #                   type=string)

def main():
    train_df_Class = Annotation(test_size=0.1,output_dataframe='0.1')
    train_df = train_df_Class.dataframe()
    test_df_Class = Annotation(test_size=0.1,output_dataframe='test')
    test_df = test_df_Class.dataframe()
    svm_model = Model(train=train_df,test=test_df)
    hyperOpt_class = controller(svm_model)
    hyperOpt_class.optimize_hyperparam()
    



if __name__ == '__main__':
    main()

