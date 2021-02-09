#!/usr/bin/env python
# coding: utf-8

# #### Loading datasets and libraries we are going to use 

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
from sklearn.metrics import mean_squared_error, accuracy_score
import sys
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
data=pd.read_csv("./data_transformed.csv")
data.describe()




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
        # which takes as parameters
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
           X, y, test_size=0.65, random_state=42)
        
    def train(self, algorithm=svm):
        
        # we set up a SVM classifier with default parameters
        self.classifier = svm.SVC(
            C=1.0, kernel='rbf', degree=3, gamma='scale')
        self.classifier.fit(self.X_train, self.y_train)
        
    def predict(self, input_data):
        return self.classifier.predict(input_data)
        
    def get_accuracy(self):
        
        # use our X_test and y_test values generated when we used
        # `train_test_split` to test accuracy.
        
        return self.classifier.score(X=self.X_test, y=self.y_test)
    
    def run_pipeline(self):
        """Helper method to run multiple pipeline methods with one call."""
        self.load_dataset()
        self.train()


# In[26]:


class PipelineFeatureEngineering(Pipeline):
    def __init__(self):
        super().__init__()
        
        # scaler to standardize the variables in the dataset
        self.scaler = StandardScaler()
        # Train the scaler once upon pipeline instantiation:
        # Compute the mean and standard deviation based on the training data
        self.scaler.fit(self.X_train)
    
    def apply_scaler(self):
        # Scale the test and training data to be of mean 0 and of unit variance
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def predict(self, input_data):
        # apply scaler transform on inputs before predictions
        scaled_input_data = self.scaler.transform(input_data)
        return self.model.predict(scaled_input_data)
                  
    def run_pipeline(self):
        """Helper method to run multiple pipeline methods with one call."""
        self.load_dataset()
        self.apply_scaler()  # updated in the this class
        self.train()


# In[27]:


data['Class'].value_counts() 


# <p>
# <strong>We perform  Unit Test</strong><br>
# We will test a few different tests for model prediction quality:


# In[28]:


class TestPredictions(unittest.TestCase):
    def setUp(self):
        # We prepare both pipelines for use in the tests
        self.pipeline_v1 = Pipeline()
        self.pipeline_v2 = PipelineFeatureEngineering()
        self.pipeline_v1.run_pipeline()
        self.pipeline_v2.run_pipeline()
        
        
        self.benchmark_predictions = [1.0] * len(self.pipeline_v1.y_test)
    
    def test_accuracy_higher_than_benchmark(self):
        # Given
        benchmark_accuracy = accuracy_score(
            y_true=self.pipeline_v1.y_test,
            y_pred=self.benchmark_predictions)
        
        predictions = self.pipeline_v1.predict(self.pipeline_v1.X_test)
        
       
        actual_accuracy = accuracy_score(
            y_true=self.pipeline_v1.y_test,
            y_pred=predictions)
        
       
        print(f'model accuracy: {actual_accuracy}, benchmark accuracy: {benchmark_accuracy}')
        self.assertTrue(actual_accuracy > benchmark_accuracy)
        
    def test_accuracy_compared_to_previous_version(self):
        
        v1_accuracy = self.pipeline_v1.get_accuracy()
        v2_accuracy = self.pipeline_v2.get_accuracy()
        print(f'pipeline v1 accuracy: {v1_accuracy}')
        print(f'pipeline v2 accuracy: {v2_accuracy}')
        
       
        self.assertTrue(v2_accuracy >= v1_accuracy)


# In[29]:


suite = unittest.TestLoader().loadTestsFromTestCase(TestPredictions)
unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)







