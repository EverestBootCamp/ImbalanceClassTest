#!/usr/bin/env python
# coding: utf-8


# In[2]:


# -*- coding: utf-8 -*-
import os
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from termcolor import colored as cl
np.random.seed(31337)


# In[3]:


data=pd.read_csv("./data_transformed.csv")
data.head()


# In[4]:


data.describe()


# In[5]:


# Decorators
from functools import wraps


def my_logger(orig_func):
    import logging
    logging.basicConfig(filename='{}.log'.format(orig_func.__name__), level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper


def my_timer(orig_func):
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper


# In[6]:


def load_data():
    data = pd.read_csv("./data_transformed.csv")
    X = data.drop('Class', axis = 1).values
    y = data['Class'].values
    return(X,y)
   # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
   # X, y, test_size=0.65, random_state=42)


# In[7]:


class Normalize(object): 
   def normalize(self, X_train, X_test):
       self.scaler = MinMaxScaler()
       X_train = self.scaler.fit_transform(X_train)
       X_test  = self.scaler.transform(X_test)
       return (X_train, X_test) 
   
   def inverse(self, X_train, X_val, X_test):
       X_train = self.scaler.inverse_transform(X_train)
       X_test  = self.scaler.inverse_transform(X_test)
       return (X_train, X_test)   

def split(X,y, splitRatio):
   X_train = X[:splitRatio]
   y_train = y[:splitRatio]
   X_test = X[splitRatio:]
   y_test = y[splitRatio:]
   return (X_train, y_train, X_test, y_test) 


# In[8]:


class TheAlgorithm(object):

  @my_logger
  @my_timer
  def __init__(self, X_train, y_train, X_test, y_test): 
      self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test    
      
  @my_logger
  @my_timer
  def fit(self): 
      normalizer = Normalize()
      self.X_train, self.X_test = normalizer.normalize(self.X_train, self.X_test)   
      train_samples = self.X_train.shape[0]
      self.classifier = svm.SVC(
          C=1.0, kernel='rbf', degree=3, gamma='scale')
      self.classifier.fit(self.X_train, self.y_train)
      self.train_y_predicted = self.classifier.predict(self.X_train)
      self.train_accuracy = np.mean(self.train_y_predicted.ravel() == self.y_train.ravel()) * 100
      self.train_confusion_matrix = confusion_matrix(self.y_train, self.train_y_predicted)        
      return self.train_accuracy
  
  @my_logger
  @my_timer
  def predict(self):
      self.test_y_predicted = self.classifier.predict(self.X_test) 
      self.test_accuracy = np.mean(self.test_y_predicted.ravel() == self.y_test.ravel()) * 100 
      self.test_confusion_matrix = confusion_matrix(self.y_test, self.test_y_predicted)        
      self.report = classification_report(self.y_test, self.test_y_predicted)
      print("Classification report for classifier:\n %s\n" % (self.report))
      return self.test_accuracy


# In[9]:


#The solution
if __name__ == '__main__': 
    X,y = load_data()
    print ('data:', X.shape, y.shape)
  
    splitRatio = 60000
    X_train, y_train, X_test, y_test = split(X,y,splitRatio) 

    np.random.seed(31337)
    ta = TheAlgorithm(X_train, y_train, X_test, y_test)
    
    train_accuracy = ta.fit()
    print()
    print('Train Accuracy:', train_accuracy,'\n') 
    print("Train confusion matrix:\n%s\n" % ta.train_confusion_matrix)
  
    test_accuracy = ta.predict()
    print()
    print('Test Accuracy:', test_accuracy,'\n') 
    print("Test confusion matrix:\n%s\n" % ta.test_confusion_matrix)
    
  
  


# In[11]:


class TestInput(unittest.TestCase):
  
    @classmethod
    def setUpClass(cls):
        # print('setupClass')   
        pass

    @classmethod
    def tearDownClass(cls): 
        # print('teardownClass')
        pass

    def setUp(self):
        print('setUp') 
        
        X, y = load_data()
        splitRatio = 60000
        self.X_train, self.y_train, self.X_test, self.y_test = split(X,y,splitRatio) 
        self.train_accuracy = 72.92166666666667
        self.train_confusion_matrix = np.array([[5447,   5,  40,  31,  49,  16, 198,  50,  81,   6],
                                                 [   3,6440, 127,  54,   3,  29,  25,  36,  24,   1],
                                                 [ 297, 420,3824, 163, 256,  19, 622, 186, 121,  50],
                                                 [ 124, 221, 255,4566,  54, 251,  97, 129, 275, 159],
                                                 [ 104, 128,  26,  54,4546, 342, 206, 133,  96, 207],
                                                 [ 399, 200, 109,1081, 416,2227, 289, 363, 228, 109],
                                                 [ 173,  89, 112,  55, 156, 229,5034,  25,  45,   0],
                                                 [ 213, 192, 205,  39, 160,  17,  26,5058,  60, 295],
                                                 [  67, 690, 202, 677,  73, 188, 347,  39,3437, 131],
                                                 [ 164, 162,  63, 290, 669, 279, 122, 735, 291,3174]])
        self.test_accuracy = 73.4
        self.test_confusion_matrix = np.array([[ 923,   1,   2,   3,   3,   1,  35,   3,   9,   0],
                                                [   0,1084,  23,  11,   0,   0,   5,   4,   8,   0],
                                                [  63,  78, 669,  27,  38,   2,  97,  28,  24,   6],
                                                [  20,  27,  35, 770,   8,  42,  18,  27,  45,  18],
                                                [  15,  21,   3,   8, 750,  60,  45,  23,  18,  39],
                                                [  56,  24,  15, 193,  73, 362,  56,  58,  38,  17],
                                                [  35,  10,  18,  11,  28,  42, 799,   6,   8,   1],
                                                [  23,  40,  52,   6,  21,   4,   7, 821,   8,  46],
                                                [  14,  90,  29,  99,  10,  33,  66,   7, 598,  28],
                                                [  21,  27,  10,  37, 133,  42,  27, 100,  48, 564]])

    def tearDown(self):
        # print('tearDown')
        pass
        
    def test_fit(self):     
        np.random.seed(31337)
        self.ta = TheAlgorithm(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertTrue(self.ta.fit(), self.train_accuracy) 
        self.assertTrue(self.ta.train_confusion_matrix.tolist(), self.train_confusion_matrix.tolist())  
  
    def test_predict(self):
        np.random.seed(31337)
        self.ta = TheAlgorithm(self.X_train, self.y_train, self.X_test, self.y_test)
        self.ta.fit()
        self.assertTrue(self.ta.predict(), self.test_accuracy)
        self.assertTrue(self.ta.train_confusion_matrix.tolist(), self.train_confusion_matrix.tolist()) 
      
      
if __name__ == '__main__':
  
    #run tests 
    unittest.main(argv=['first-arg-is-ignored'], exit=False)







