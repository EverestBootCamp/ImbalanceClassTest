
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

    File path where the output csv file need to be saved. Input should be in String format.
    output_file_path='C:\\Users\\udayr\\OneDrive\\Desktop'
    '''

    def __init__(
            self,
            input_file_path='data_transformed.csv',
            target_column='Class',
            test_size=0.3,
            output_file_path='C:\\Users\\udayr\\OneDrive\\Desktop'):
        self.input_file_path = input_file_path
        self.target_column = target_column
        self.test_size = test_size
        self.output_file_path = output_file_path

    def data_as_csv(self):

        # Read the data_transformed file
        df = pd.read_csv(self.input_file_path)

        # Splitting the independednt adn dependent variable
        X = df.drop(columns=[self.target_column])  # input
        y = df[self.target_column]  # output

        # Splitting 70% of the data for training and 30% for testing (with
        # stratification for imbalanced class)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=0)

        # making a copy to avoid SettingWithCopy Warning
        train_data = X_train.copy()
        test_data = X_test.copy()

        # Combining the stratified Train nad Test data separatley
        train_data.loc[:, self.target_column] = y_train
        test_data.loc[:, self.target_column] = y_test

        # Setting the names for Train and test data csv files, of the full data
        train_dataset_name = self.output_file_path + \
            '\\data_transformed_for_training_' + str(70) + "pct.csv"
        test_dataset_name = self.output_file_path + \
            '\\data_transformed_for_testing_' + str(30) + "pct.csv"

        # Saving Train and test data csv files, of the full data
        train_data.to_csv(train_dataset_name, index=False)
        print('{} saved'.format(train_dataset_name))
        test_data.to_csv(test_dataset_name, index=False)
        print('{} saved'.format(test_dataset_name))

        # Since the startified 70 ercent trainig data is actually 100 percent
        # annotated data. So, saving the same file as the 100 percent annotated
        # file.
        dataset_name = self.output_file_path + '\\data_transformed_annotated_100_pct.csv'
        train_data.to_csv(dataset_name, index=False)
        print('Stratified 70% Training data re-saved - {}'.format(dataset_name))

        # Read the Stratified 70% data_transformed file
        df_70 = pd.read_csv(train_dataset_name)

        # Splitting the independednt adn dependent variable
        X_70 = df_70.drop(columns=[self.target_column])  # input
        y_70 = df_70[self.target_column]  # output

        for size in list(np.linspace(0.1, 0.9, 9)):
            '''
            Performing Train test split on the stratified 70% data and in the loop incresing the test size by 10%
            with each step. Then combine only the Test data, to have the data in 10% advancing portions and maintain the
            category proprtion in the Class feature. The output will be the saved csv files of the combined Test data.
            '''
            _, Sample_X, _, Sample_y = train_test_split(
                X_70, y_70, test_size=self.test_size, stratify=y_70, random_state=0)
            dataset_name = self.output_file_path + \
                '\\data_transformed_annotated_' + str(round(size * 100)) + "pct.csv"
            train_data = Sample_X.copy()
            train_data.loc[:, 'Class'] = Sample_y
            train_data.to_csv(dataset_name, index=False)
            print('{} saved'.format(dataset_name))
