import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
class Model():
 # reading train and test data and scaling them
 # CV=k value in cross validation technique
  def __init__(self,train_path,test_path,CV=5,label='Class'):
    train = pd.read_csv(train_path) #reading train data
    test = pd.read_csv(test_path) #reading test data
    self.X_train= train.iloc[:,:-1]
    self.X_test= test.iloc[:,:-1]
    self.y_train = train.iloc[:,:][label] 
    self.y_test = test.iloc[:,:][label]
    sc = StandardScaler() #scaling data
    self.scaled_X_train = sc.fit_transform(self.X_train)
    self.scaled_X_test = sc.transform(self.X_test)
    self.CV=CV

    #definig the SVM model and corresponding parameters
        #At the end, f1 score of model will be calculated
    # hyper parameters: type of kernel, C, gamma,..
  def svm(self,hyper_parameter={}):
    
      clf = svm.SVC(**hyper_parameter, class_weight = 'balanced',\
                        random_state=0)
      clf.fit(self.scaled_X_train, self.y_train)
      cv_mean = np.mean(cross_val_score(clf, self.scaled_X_train, self.y_train, cv=self.CV))
      cv_std = np.std(cross_val_score(clf, self.scaled_X_train, self.y_train, cv=self.CV))
      self.y_pred = clf.predict(self.scaled_X_test)
      return f1_score(self.y_test,self.y_pred)
# applying model on data file
model =Model("data_transformed.csv","data_transformed.csv")
# applting SVM on data
model.svm(hyper_parameter={'kernel':'linear','C':1.0,'gamma' : 'auto'})