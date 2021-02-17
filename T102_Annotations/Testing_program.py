'''
Required Liraries Imported
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import collections


from T102_Annotations import annotation

#Read the data_transformed file
df = pd.read_csv("data_transformed.csv")           
df

a = annotation(df)  #Calling the class and assigning it to 'a' with the dataset as the argument

a.data_as_csv()