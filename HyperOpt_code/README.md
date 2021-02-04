# Analysis Goal :
To perform hyper-parameter tuning using HyperOpt for SVM model

## Hyperparameter Tuning with hyperopt
Hyperparameter tuning is an important step for maximizing the performance of a model. Hyperparameters are certain values/weights that determine the learning process of an algorithm. Several Python packages have been developed specifically for this purpose. Scikit-learn provides a few options, GridSearchCV and RandomizedSearchCV being two of the more popular options. Outside of scikit-learn, the Optunity, Spearmint and hyperopt packages are all designed for optimization. In this task, we will focus on the hyperopt package.

# Conclusion:
This task is done in a team of 2 students. The given dataset was analyzed and modelled using SVM Model. Hyperparameters were tuned using hyperopt. Hyperparameter tuning is an important step in building a learning algorithm model. Best parameters for SVM model are 'C': 4.250168671220421, 'gamma': 0.01953647948588498, 'kernel': 'rbf'. Modelled SVM with these hyperparameters. For class==0 we got all values precision, recall, f1-score as 1.00. For Class==1 precision is 0.97, recall is 0.77, f1-score is 0.86 which is pretty good. Recall can be thought of as a measure of classifier completeness. With the help of the pickle library, we got 0.9995602213079869 accuracy of the SVM model on unseen data.
