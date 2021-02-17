from tpot import TPOTClassifier
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# makesure target class lebel column name in the dataset is 'Class'
class Tpot:

    def __init__(self, datafile):
        self.data_frame = pd.read_csv(datafile)

    def run_tpot(
        self,
        fullyqualified_op_filename,
        tpot_generations,
        tpot_population_size,
        tpot_maxtime,
        cv_k_fold,
        cv_scoring):

        df_without_class = self.data_frame.drop(
            ["Class"], axis=1
        )
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer = imputer.fit(
            df_without_class
        )
        # apply imputer class to remove anomalies in the dataset
        impute_df = imputer.transform(df_without_class)
        X = pd.DataFrame(impute_df, columns=df_without_class.columns)
        y = self.data_frame["Class"]

        # tpot config can be modified as required
        tpot_config = {
            "sklearn.preprocessing.Binarizer": {
                "threshold": np.arange(0.0, 1.01, 0.05)
            },
            "sklearn.decomposition.FastICA": {"tol": np.arange(0.0, 1.01, 0.05)},
            "sklearn.cluster.FeatureAgglomeration": {
                "linkage": ["ward", "complete", "average"],
                "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"],
            },
            "sklearn.preprocessing.MinMaxScaler": {},
            "sklearn.preprocessing.Normalizer": {"norm": ["l1", "l2", "max"]},
            "sklearn.kernel_approximation.Nystroem": {
                "kernel": [
                    "rbf",
                    "cosine",
                    "chi2",
                    "laplacian",
                    "polynomial",
                    "poly",
                    "linear",
                    "additive_chi2",
                    "sigmoid",
                ],
                "gamma": np.arange(0.0, 1.01, 0.05),
                "n_components": range(1, 11),
            },
            "sklearn.decomposition.PCA": {
                "svd_solver": ["randomized"],
                "iterated_power": range(1, 11),
            },
            "sklearn.preprocessing.PolynomialFeatures": {
                "degree": [2],
                "include_bias": [False],
                "interaction_only": [False],
            },
            "sklearn.kernel_approximation.RBFSampler": {
                "gamma": np.arange(0.0, 1.01, 0.05)
            },
            "sklearn.preprocessing.RobustScaler": {},
            "sklearn.preprocessing.StandardScaler": {},
            "tpot.builtins.ZeroCount": {},
            "tpot.builtins.OneHotEncoder": {
                "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
                "sparse": [False],
            },
            "sklearn.svm.LinearSVC": {},
            "sklearn.neighbors.KNeighborsClassifier": {},
            "sklearn.linear_model.LogisticRegression": {},
            "sklearn.ensemble.RandomForestClassifier": {},
            "sklearn.ensemble.GradientBoostingClassifier": {},
            "sklearn.tree.DecisionTreeClassifier": {},
            "xgboost.XGBClassifier": {},
        }

        # apply tpot
        tpot = TPOTClassifier(
            generations=tpot_generations,
            population_size=tpot_population_size,
            max_time_mins=tpot_maxtime,
            verbosity=2,
            cv=cv_k_fold,
            scoring=cv_scoring,
            config_dict=tpot_config
        )
        tpot.fit(X, y)
        winning_pipeline = tpot.fitted_pipeline_
        print("winning tpot pipeline is:", winning_pipeline)

        if fullyqualified_op_filename is not None:
            tpot.export(fullyqualified_op_filename)

    #Fullyqualified_op_filename with path along with additional '\' is
    #required to generate python file for winning pipeline. for example:
    #Fullyqualified_op_filename=
    #"C:\\Users\\SG\\Desktop\\lantern\\local_notebook\\winning_pipeline.py"
    # tpot_maxtime is max time to run tpot default is set to 60 mins
    def apply_tpot(self,
        fullyqualified_op_filename=None,
        tpot_generations=5,
        tpot_population_size=20,
        tpot_maxtime=720,
        cv_k_fold=5,
        cv_scoring="balanced_accuracy"):
        try:
            self.run_tpot(
                fullyqualified_op_filename,
                tpot_generations,
                tpot_population_size,
                tpot_maxtime,
                cv_k_fold,
                cv_scoring
            )
        except AssertionError as error:
            print(error)
