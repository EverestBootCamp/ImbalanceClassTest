#!/usr/bin/env python
# coding: utf-8

# #### Loading datasets 

# In[18]:


from sklearn import datasets 
import pandas as pd 
import numpy as np
from termcolor import colored as cl
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import unittest
import sys
data=pd.read_csv("./data_transformed.csv")
data.describe()


# <strong>Create the Pipelines</strong>

# In[22]:


class Pipeline:
    def __init__(self):
    
        # None when the class is instantiated.
        self.X_train, self.X_test, self.y_train, self.Y_test = None, None, None, None
        self.model = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the dataset and perform train test split."""
        data = pd.read_csv("./data_transformed.csv")
        X = data.drop('Class', axis = 1).values
        y = data['Class'].values
        
        # we divide the data set using the train_test_split function from sklearn, 
        # which takes as parameters, the dataframe with the predictor variables, 
        # then the target, then the percentage of data to assign to the test set, 
        # and finally the random_state to ensure reproducibility.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
           X, y, test_size=0.65, random_state=42)
        
    def train(self, algorithm=svm):
        
        # we set up a SVM classifier with default parameters
        self.classifier = svm.SVC(
            C=1.0, kernel='rbf', degree=3, gamma='scale')
        self.classifier.fit(self.X_train, self.y_train)
        
    def predict(self, input_data):
        return self.model.predict(input_data)
        
    def get_accuracy(self):
        
        # use our X_test and y_test values generated when we used
        # `train_test_split` to test accuracy.
        # score is a method on the Logisitic Regression that 
        # returns the accuracy by default, but can be changed to other metrics, see: 
        return self.classifier.score(X=self.X_test, y=self.y_test)
    
    def run_pipeline(self):
        """Helper method to run multiple pipeline methods with one call."""
        self.load_dataset()
        self.train()


# In[23]:


class PipelineFeatureEngineering(Pipeline):
    def __init__(self):
        super().__init__()
        
        # scaler to standardize the variables in the dataset
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
    
    def apply_scaler(self):
        # Scale the test and training data 
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def predict(self, input_data):
        # apply scaler transform on inputs 
        scaled_input_data = self.scaler.transform(input_data)
        return self.model.predict(scaled_input_data)
                  
    def run_pipeline(self):
        """Helper method to run multiple pipeline methods with one call."""
        self.load_dataset()
        self.apply_scaler() 
        self.train()


# In[24]:


pipeline = PipelineFeatureEngineering()
pipeline.run_pipeline()
accuracy_score = pipeline.get_accuracy()
print(f'current model accuracy is: {accuracy_score}')


# In[25]:


class TestDataEngineering(unittest.TestCase):
    def setUp(self):
        self.pipeline = PipelineFeatureEngineering()
        self.pipeline.load_dataset()
    
    def test_scaler_preprocessing_brings_x_train_mean_near_zero(self):
        # convert the dataframe to be a single column with pandas stack
        original_mean = self.pipeline.X_train.mean()
        self.pipeline.apply_scaler()
        
        # The  StandardScaler is that it will transform your data 
        # to center the distribution at 0 and scale the variance at 1.
       
        self.assertTrue(original_mean > self.pipeline.X_train.mean())  
        self.assertAlmostEqual(self.pipeline.X_train.mean(), 0.0, places=3)
        print(f'Original X train mean: {original_mean}')
        print(f'Transformed X train mean: {self.pipeline.X_train.mean()}')
        
    def test_scaler_preprocessing_brings_x_train_std_near_one(self):
        self.pipeline.apply_scaler()
        self.assertAlmostEqual(self.pipeline.X_train.std(), 1.0, places=3)
        print(f'Transformed X train standard deviation : {self.pipeline.X_train.std()}')


# In[26]:


suite = unittest.TestLoader().loadTestsFromTestCase(TestDataEngineering)
unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)







