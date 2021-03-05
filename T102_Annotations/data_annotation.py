
'''
Required Liraries Imported
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


'''
Required Liraries Imported
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Annotation():
    '''
    Following arguments are required to run this annotation class.Default values are provided:-

    File path where the original data csv file is saved. Input should be in String format.
    input_file_path='data_transformed.csv'

    The dependent or target feature. Input should be in String format.
    target_column='Class'

    Test size for the train test split. Input should be in integer format.
    test_size=0.3

    The type of dataframe can be selected by iputting the following arguments:-
    'train' or 'test' or Any float value ranging from 0-1
    output_dataframe='train'
    '''

    def __init__(
            self,
            input_file_path='data_transformed.csv',
            target_column='Class',
            test_size=0.3,
            output_dataframe='train'):
        self.input_file_path = input_file_path
        self.target_column = target_column
        self.test_size = test_size
        self.output_dataframe = output_dataframe

    def dataframe(self):

        # Read the data_transformed file
        df = pd.read_csv(self.input_file_path)

        # Splitting the independednt adn dependent variable
        X = df.drop(columns=[self.target_column])  # input
        y = df[self.target_column]  # output

        # Splitting 70% of the data for training and 30% for testing (with
        # stratification for imbalanced class)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=0)
        
        if self.output_dataframe.isalpha() == True:
            if self.output_dataframe.lower() == 'train':
                # making a copy to avoid SettingWithCopy Warning
                train_data = X_train.copy()
                # Combining the stratified data
                train_data.loc[:, self.target_column] = y_train
                return train_data
            elif self.output_dataframe.lower() == 'test':
                # making a copy to avoid SettingWithCopy Warning
                test_data = X_test.copy()
                # Combining the stratified data
                test_data.loc[:, self.target_column] = y_test
                return test_data
        elif self.output_dataframe.isalpha() == False:
            if float(self.output_dataframe) in np.linspace(0.1, 0.9, 9):
                # making a copy to avoid SettingWithCopy Warning
                train_data = X_train.copy()
                # Combining the stratified data
                train_data.loc[:, self.target_column] = y_train

                # Splitting the independednt adn dependent variable
                X_70 = train_data.drop(columns=[self.target_column])  # input
                y_70 = train_data[self.target_column]  # output
                _, Sample_X, _, Sample_y = train_test_split(
                    X_70, y_70, test_size=float(self.output_dataframe), stratify=y_70, random_state=0)

                # making a copy to avoid SettingWithCopy Warning
                train_data = Sample_X.copy()
                # Combining the stratified data
                train_data.loc[:, self.target_column] = Sample_y
                return train_data
            else:
                print("""For the 'output_dataframe' argument, select from these:-
                  'train' or 'test' or Any float value ranging from 0-1""")
        else:
            print("""For the 'output_dataframe' argument, select from these:-
                  'train' or 'test' or Any float value ranging from 0-1""")
