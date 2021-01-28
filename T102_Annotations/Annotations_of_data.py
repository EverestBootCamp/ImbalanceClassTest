class Annotations_of_Data():
    def __init__(self,df):
        self.df = df
        
    def stratified_train_test_split(self):
        
        '''
        Required Liraries Imported
        '''
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        import collections, numpy
        
        def get_class_counts(y):
            '''
            picks up the dependet variable as an argument,
            to get count of it's occurences in the main data
            '''
            print('Count for 0 in the data = {}'.format(collections.Counter(y)[0]))
            print('Count for 1 in the data = {}'.format(collections.Counter(y)[1]))
        
        def get_class_proportions(y):
            '''
            picks up the dependet variable as an argument,
            to get proportion of it's occurences in the main data
            '''
            print('Proportion for 0 in the data = {}'.format(collections.Counter(y)[0]/(collections.Counter(y)[0] + collections.Counter(y)[1])))
            print('Proportion for 1 in the data = {}'.format(collections.Counter(y)[1]/(collections.Counter(y)[0] + collections.Counter(y)[1])))
        
        i = 0.1
        while i<=1:
            '''
            Feeding the train and test split with 10% of data in starting
            and addition of 10% data with each loop run.
            Once whole data is feeded, the loop breaks.
            '''
            df_parts = df.sample(frac=i, random_state=42)
            print('Shape of the {} fraction of the data ={}'.format(i, df_parts.shape))

            X = df_parts.iloc[:,:-1] #input, independent features
            y = df_parts['Class']    #output, dependent features
            #Print the shape of X & Y
            print('Shape of the {} fraction of the independent features ={}'.format(i, X.shape))
            print('Shape of the {} fraction of the dependent features ={}'.format(i, y.shape))

            # Implementing stratified train and test split with 70% train and 30% test size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify= y, random_state = 0)

            get_class_counts(y_train)
            get_class_proportions(y_train)

            get_class_counts(y_test)
            get_class_proportions(y_test)
            print("*******************************************************************************************************")
            i += 0.1

            if i>1:
                break