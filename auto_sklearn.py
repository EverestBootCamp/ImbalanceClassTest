import autosklearn.classification
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import ast

class Autosklearn:
 
    
    def __init__(self, datafile):
        self.df = pd.read_csv(datafile)

    def run_autosklearn(
        self,
        Fullyqualified_op_filename,
        maxtime,
        per_run_time_limit,                  
        scoring):
        
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
        
       # apply autosklearn
        autosklearn_est = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=maxtime,
        per_run_time_limit=per_run_time_limit,
        include_estimators=["random_forest", "decision_tree", "libsvm_svc", ], #,"decision_tree", "gradient_boosting", "libsvm_svc",
        include_preprocessors=["no_preprocessing", ],
        scoring_functions=[scoring])#balanced_accuracy, precision, recall,"f1"
        autosklearn_est.fit(X, y)   
   
        #winning_pipeline        
        losses_and_configurations = [(run_value.cost, run_key.config_id)
        for run_key, run_value in automl.automl_.runhistory_.data.items()]
        losses_and_configurations.sort()
        print("Lowest loss:", losses_and_configurations[0][0])
        print("Best configuration:", automl.automl_.runhistory_.ids_config[losses_and_configurations[0][1]])
        
        '''can be printed when scoring function is not specified'''
        #print("autosklearn statistics: ", automl.sprint_statistics())
        #print("autosklearn cv result  is:", autosklearn_est.cv_results_)
        #print("autosklearn model is:", autosklearn_est.show_models())
        


    # Fullyqualified_op_filename with path along with additional '\' is required to generate python file for winning pipeline. for example Fullyqualified_op_filename="C:\\Users\\SG\\Desktop\\lantern\\local_notebook\\winning_pipeline.py"
    # tpot_maxtime is max time to run tpot default is set to 60 mins
    def apply_autosklearn(self,
        Fullyqualified_op_filename=None,
        maxtime=7200,
        per_run_time_limit=1800,                  
        scoring="f1"#balanced_accuracy, precision, recall
        ):
        try:
            self.run_autosklearn(
                Fullyqualified_op_filename,
                maxtime,
                per_run_time_limit,
                scoring
            )
        except AssertionError as error:
            print(error)
