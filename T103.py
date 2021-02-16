import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
class Model:

  def __init__(self,datafile):
    self.data = pd.read_csv(datafile)
    self.data = self.data.rename(columns={"Unnamed: 0": "T0"})
        
     # defining function for spliting data to Train and test portions
  def split(self, test_size):
    X = self.data.iloc[:1000,:-1]
    y = self.data.iloc[:1000,:]['Class'] 
    self.X_train, self.X_test, self.y_train, self.y_test\
    = train_test_split(X, y, test_size = test_size,\
                       stratify= y, random_state = 0)

    # defining function for scaling data 
  def scaler(self):
    sc = StandardScaler()
    self.scaled_X_train = sc.fit_transform(self.X_train)
    self.scaled_X_test = sc.transform(self.X_test)

    #definig the SVM model and corresponding parameters
  def svm(self,kernel_type,hyper_parameter={}):
    
    if kernel_type == 'linear':
      hyper_parameter = hyper_parameter or {'C': 1.0}
      clf = svm.SVC(kernel= kernel_type, class_weight =\
                    'balanced', C = hyper_parameter.get('C'),\
                    random_state=0)

    if kernel_type == 'poly':
      hyper_parameter = hyper_parameter or { 'C': 1.0,\
                                            'degree': 3, 'coef0': 0  }
      clf = svm.SVC(kernel=kernel_type, class_weight =\
                    'balanced', degree = hyper_parameter.get('degree'),\
                    C = hyper_parameter.get('C'),\
                    coef0 = hyper_parameter.get('coef0'), random_state=0)

    if kernel_type == 'rbf':
      hyper_parameter = hyper_parameter or {'C': 1.0,\
                                            'gamma' : 'scale' }
      clf = svm.SVC( kernel=kernel_type, class_weight = 'balanced',\
                    gamma = hyper_parameter.get('gamma'),\
                    C = hyper_parameter.get('C'),random_state=0)
  
    if kernel_type == 'sigmoid':
      hyper_parameter = hyper_parameter or {'C': 1.0 ,\
                                            'gamma' : 'scale',\
                                            'coef0': 0 }
      clf = svm.SVC(kernel=kernel_type, class_weight = 'balanced',\
                    gamma = hyper_parameter.get('gamma'),\
                    C = hyper_parameter.get('C'), \
                    coef0 = hyper_parameter.get('coef0'), random_state=0)
   
    clf.fit(self.scaled_X_train, self.y_train)
    self.y_pred = clf.predict(self.scaled_X_test)
    
    #defining function for evaluating the performance of the model
  def performance(self):
    cm = confusion_matrix(self.y_test, self.y_pred)
    ac = accuracy_score(self.y_test, self.y_pred)
    print("Confusion Matrix: \n", cm)
    print("Accuracy:",ac)

# applying model on data file
model = Model(datafile= "../data_transformed.csv")
# spliting data with predefined split function in the class
model.split(test_size=0.25)
# scaling data with predefined scaler function in the class
model.scaler()
# applting SVM on data
model.svm(kernel_type="rbf", hyper_parameter = {'C': 1.0,'gamma' : 'scale'})