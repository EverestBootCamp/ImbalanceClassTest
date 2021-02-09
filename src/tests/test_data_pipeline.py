#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
from termcolor import colored as cl
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import unittest
import sys
import pytest

# Read the data from dataset
df = pd.read_csv("./data_transformed.csv")
df.head()


# In[73]:


#df['feature_names'] = list(range(len(df.index)))
df.describe()


# In[74]:


class SimplePipeline:
    def __init__(self):
        
        # None when the class is instantiated.
        self.X_train, self.X_test, self.y_train, self.Y_test = None, None, None, None
        self.model = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the dataset and perform train test split."""
        
        df = pd.read_csv("./data_transformed.csv")
        df['feature_names'] = list(range(len(df.index))) 
        X=df.drop('Class',axis=1).values
        y = df['Class'].values
        
        # we divide the data set using the train_test_split function from sklearn, 
        # which takes as parameters, the dataframe with the predictor variables, 
        # then the target, then the percentage of data to assign to the test set, 
        # and finally the random_state to ensure reproducibility.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
          X, y, test_size=0.65, random_state=42)
        
    def train(self, algorithm=svm):
        
        # we set up a SVM classifier with default parameters
        self.model = svm.SVC(
            C=1.0, kernel='rbf', degree=3, gamma='scale')
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, input_data):TechnicalMentorship
        return self.model.predict(input_data)
        
    def get_accuracy(self):
        
        # use our X_test and y_test values generated when we used
        # `train_test_split` to test accuracy.
        # score is a method on the Logisitic Regression that 
        # returns the accuracy by default, but can be changed to other metrics, see: 
        
        return self.model.score(X=self.X_test, y=self.y_test)
    
    def run_pipeline(self):
        """Helper method to run multiple pipeline methods with one call."""
        self.load_dataset()
        self.train()


# In[75]:


pipeline = SimplePipeline()
pipeline.run_pipeline()
accuracy_score = pipeline.get_accuracy()

# note that f' string interpolation syntax requires python 3.6
# https://www.python.org/dev/peps/pep-0498/
print(f'current model accuracy is: {accuracy_score}')







