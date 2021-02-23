from tpot import TPOTClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
import ast

class Tpot:
    
    def __init__(self, datafile):
        self.df = pd.read_csv(datafile)

    def run_tpot(
        self,
        Fullyqualified_op_filename,
        tpot_generations,
        tpot_population_size,
        tpot_maxtime,
        CV_K_fold,
        cv_scoring,
        tpot_config):
        
        df_without_class = self.df.drop(
            ["Class"], axis=1
        )  # makesure target class lebel column name in the dataset is 'Class'

        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer = imputer.fit(
            df_without_class
        )  # apply imputer class to remove anomalies in the dataset
        impute_df = imputer.transform(df_without_class)
        X = pd.DataFrame(impute_df, columns=df_without_class.columns)
        y = self.df["Class"]
        
        #open tpot config file
        if tpot_config != None:
            with open(tpot_config) as file:
                data= file.read()
                tpot_config = ast.literal_eval(data)
  
        # apply tpot
        tpot = TPOTClassifier(
            generations=tpot_generations,
            population_size=tpot_population_size,
            max_time_mins=tpot_maxtime,
            verbosity=2,
            cv=CV_K_fold,
            scoring=cv_scoring,
            config_dict=tpot_config
        )
        tpot.fit(X, y)
        winning_pipeline = tpot.fitted_pipeline_
        print("winning tpot pipeline is:", winning_pipeline)

        if Fullyqualified_op_filename != None:
            tpot.export(Fullyqualified_op_filename)

    # Fullyqualified_op_filename with path along with additional '\' is required to generate python file for winning pipeline. for example Fullyqualified_op_filename="C:\\Users\\SG\\Desktop\\lantern\\local_notebook\\winning_pipeline.py"
    # tpot_maxtime is max time to run tpot default is set to 60 mins
    def apply_tpot(self,
        Fullyqualified_op_filename=None,
        tpot_generations=5,
        tpot_population_size=20,
        tpot_maxtime=720,
        CV_K_fold=5,
        cv_scoring="balanced_accuracy",
        tpot_config= None):
        try:
            self.run_tpot(
                Fullyqualified_op_filename,
                tpot_generations,
                tpot_population_size,
                tpot_maxtime,
                CV_K_fold,
                cv_scoring,
                tpot_config
            )
        except AssertionError as error:
            print(error)
