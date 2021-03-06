#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from hyperopt import space_eval
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


# In[2]:


class controller():
    def _init_(self,model):
        self.model=model
    def objective(space4svm):
            f1 = self.model.svm(space4svm)
            print(f1)
            return {'loss': -f1, 'status': STATUS_OK}
    space4svm ={
              'C': hp.uniform('C', 0, 20),
              'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
              'gamma': hp.choice('gamma', ['scale', 'auto']),
              'degree': hp.quniform('degree', 3, 10,1),
              'coef0': hp.uniform('coef0', 0,10)
               }
    
    
    # Initialize trials object.
    trials = Trials()
    #using seed to get repeatable results.
    seed = 123
    # run the hyper paramter tuning.
    best = fmin(fn=objective,   
                        space=space4svm,
                        algo=tpe.suggest,
                        max_evals=10,
                        trials=trials,
                        rstate=np.random.RandomState(seed))
    print (space_eval(space4svm,best))



# In[ ]:




