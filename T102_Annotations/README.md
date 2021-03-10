#  T102 :Create annotations of data, train and testset

## Analysis Goal
By running this Annotations class, stratified data with increment of 10% data will be saved in CSV foormat.
Also, train and test files will be saved separately in csv formats.

## Conclusion
The arguments are required to run this annotation class. Default values are provided:-
    
    File path where the original data csv file is saved. Input should be in String format.
    input_file_path='data_transformed.csv'
    
    The dependent or target feature. Input should be in String format.
    target_column='Class'
    
    Test size for the train test split. Input should be in integer format.
    test_size=0.3
    
    File path where the output csv file need to be saved. Input should be in String format.
    output_file_path='C:\\Users\\udayr\\OneDrive\\Desktop'

To run the data_annotation class download following files to your systtem:-
- data_annotation.py  
- data_annotation_test.py

And through your terminal 'cd' to the file location and run the following command:-
* python data_annotation_test.py
