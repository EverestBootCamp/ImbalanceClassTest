{      'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': [1,2,3,4,5,6,7,8,9,10,11],
        'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': [0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,0.55, 0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],
        'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': [0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,0.55, 0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],
        'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'min_samples_leaf':  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': [1,2,3,4,5,6,7,8,9,10,11],
        'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'subsample': [0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,0.55, 0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],
        'max_features': [0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,0.55, 0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    },

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },

    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': [1,2,3,4,5,6,7,8,9,10,11],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': [0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,0.55, 0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],
        'min_child_weight': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'n_jobs': [1],
        'verbosity': [0]
    },

    'sklearn.linear_model.SGDClassifier': {
        'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    'sklearn.neural_network.MLPClassifier': {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
    },

           "sklearn.cluster.FeatureAgglomeration": {
                "linkage": ["ward", "complete", "average"],
                "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"]
            },
            "sklearn.preprocessing.MinMaxScaler": {},
            "sklearn.preprocessing.Normalizer": {"norm": ["l1", "l2", "max"]},
            "sklearn.decomposition.PCA": {
                "svd_solver": ["randomized"],
                "iterated_power": [1,2,3,4,5,6,7,8,9,10,11]
            },
            "sklearn.preprocessing.PolynomialFeatures": {
                "degree": [2],
                "include_bias": [False],
                "interaction_only": [False]
            },
            "sklearn.kernel_approximation.RBFSampler": {},
            "sklearn.preprocessing.RobustScaler": {},
            "sklearn.preprocessing.StandardScaler": {},
            "tpot.builtins.ZeroCount": {},
            "tpot.builtins.OneHotEncoder": {
                "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
                "sparse": [False]
            }
            
        }
