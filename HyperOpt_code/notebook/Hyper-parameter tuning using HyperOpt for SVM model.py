#!/usr/bin/env python
# coding: utf-8

# # Analysis Goal :

# To perform hyper-parameter tuning using HyperOpt for SVM model

# ## Hyperparameter Tuning with hyperopt

# Hyperparameter tuning is an important step for maximizing the performance of a model. Hyperparameters are certain values/weights that determine the learning process of an algorithm. Several Python packages have been developed specifically for this purpose. Scikit-learn provides a few options, GridSearchCV and RandomizedSearchCV being two of the more popular options. Outside of scikit-learn, the Optunity, Spearmint and hyperopt packages are all designed for optimization. In this task, we will focus on the hyperopt package.
# 

# ### HyperOpt:

# It is a powerful python library that search through an hyperparameter space of values . It implements three functions for minimizing the cost function,
# 
# * Random Search
# * TPE (Tree Parzen Estimators)
# * Adaptive TPE

# In[1]:


import pandas as pd
import numpy as np
import os
import string


# In[2]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


# In[3]:


np.random.seed(123)


# In[4]:


# import data
data_transformed = pd.read_csv('data_transformed.csv')

# avoid this ugly slicing by using a two-dim dataset
X = data_transformed.iloc[:,:-1]
y = data_transformed.iloc[:,:]['Class']


# using 75% of the data for training and 25% for testing (with stratification for imbalanced class)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify= y, random_state = 123)

#Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
scaled_X_train = sc.fit_transform(X_train)
scaled_X_test = sc.transform(X_test)


# In[5]:


# Defining the space for hyperparameter tuning

'''
hp.choice(label, options) — Returns one of the options, which should be a list or tuple.
hp.loguniform(label, low, high) — Returns a value drawn according to exp(uniform(low, high)) 
so that the logarithm of the return value is uniformly distributed.When optimizing, 
this variable is constrained to the interval [exp(low), exp(high)].

'''

space =  {
   'C':hp.loguniform("C", np.log(1), np.log(100)),
   'kernel':hp.choice('kernel',['rbf', 'poly']),
   'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),    
}

#SVM Parameters

'''
C — Regularization parameter. The strength of the regularization is inversely proportional to C.
kernel — Specifies the kernel type to be used in the algorithm.
gamma — Kernel coefficient for ‘rbf’, ‘poly’.

'''

# Define the function to minimize (SVM Model)
def objective(space):
    clf = svm.SVC( C = space['C'], kernel= space['kernel'], gamma = space['gamma']) 
    evaluation = [(scaled_X_train, y_train), (scaled_X_test, y_test)]
    
# Fit the model on training set    
    clf.fit(scaled_X_train, y_train)
# Make a prediction 
    pred = clf.predict(scaled_X_test)
# Calculate our Metric - accuracy
    accuracy = accuracy_score(y_test, pred)
    print ("SCORE:", accuracy )
    
# Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -accuracy, 'status': STATUS_OK}


# Initialize trials object.
trials = Trials()
#using seed to get repeatable results.
seed = 123
# run the hyper paramter tuning.
best = fmin(fn=objective,   
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials,
           rstate= np.random.RandomState(seed))

print (best)


# Here, ‘best’ gives you the optimal parameters that best fit model and better loss function value.

# ## Analyze results by using trials object

# In[6]:


trials.results


# ‘trials’, it is an object that contains or stores all the statistical and diagnostic information such as hyperparameter, loss-functions for each set of parameters that the model has been trained. ‘fmin’, it is an optimization function that minimizes the loss function and takes in 4 inputs. Algorithm used is ‘tpe.suggest’ , other algorithm that can be used is ‘tpe.rand.suggest’.

# In[7]:


from hyperopt import space_eval


# In[8]:


#using space_eval for finding best parameters
space_eval(space, best)


# In[9]:


# Model SVM with best parameters 
clf = svm.SVC( C = 20.22424088022293  , gamma = 0.005597946604318103  , kernel= 'poly') 
clf.fit(scaled_X_train, y_train)
pred = clf.predict(scaled_X_test)

#Compute confusion matrix to evaluate the accuracy of a classification.
print("confusion_matrix:")
print(confusion_matrix(y_test, pred))

#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
print("roc_auc_score:" )
print( roc_auc_score(y_test, pred))

#Build a text report showing the main classification metrics.
print("classification_report:")
print( classification_report(y_test, pred))


# . For class==0 we got all values precision, recall, f1-score as 1.00. For Class==1 precision is 0.97, recall is 0.77, f1-score is 0.86 which is pretty good.

# # Conclusion:

# This task is done in a team of 2 students. The given dataset was analyzed and modelled using SVM Model. Hyperparameters were tuned using hyperopt. Hyperparameter tuning is an important step in building a learning algorithm model. Best parameters for SVM model are 'C': 20.22424088022293, 'gamma': 0.005597946604318103, 'kernel': 'poly'. Modelled SVM with these hyperparameters. For class==0 we got all values precision, recall, f1-score as 1.00. For Class==1 precision is 0.97, recall is 0.77, f1-score is 0.86 which is pretty good. Recall can be thought of as a measure of classifier completeness.
