import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optuna
import optuna.integration.lightgbm as lgb
import math
from optuna.trial import Trial
np.random.seed(123)

# import data
data_transformed = pd.read_csv('data_transformed.csv')

# avoid this ugly slicing by using a two-dim dataset
X = data_transformed.iloc[:, :-1]
y = data_transformed.iloc[:, :]['Class']


# using 75% of the data for training and 30% for testing (with
# stratification for imbalanced class)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=123)

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
scaled_X_train = sc.fit_transform(X_train)
scaled_X_test = sc.transform(X_test)
class LGBM():
  
  def __init__(self,scaled_X_train, scaled_X_test, y_train, y_test):
    dtrain = lgb.Dataset(scaled_X_train, label= y_train)
    dtest = lgb.Dataset(scaled_X_test, label=y_test)
    
  def objective(self, trial = Trial):
    dtrain = lgb.Dataset(scaled_X_train, label= y_train)
    dtest = lgb.Dataset(scaled_X_test, label=y_test)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": 0,
        "boosting_type": "gbdt",
    }
    gbm = lgb.train(
        params, dtrain, verbose_eval=True, valid_sets=[dtest]
    )

    preds = gbm.predict(scaled_X_test)
    y_pred = np.array(list(map(lambda x: int(x), preds>0.5)))
    f1_sc = sklearn.metrics.f1_score(y_test, y_pred)
    loss = np.subtract(1,f1_sc)
    return loss

  def optuna_method(self):
    study = optuna.create_study(direction="minimize")
    study.optimize(self.objective, n_trials=2000)
    self.params = study.best_params
    return study.best_trial

