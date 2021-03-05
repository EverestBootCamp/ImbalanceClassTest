'''
Required Liraries Imported
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import collections

import data_annotation
from data_annotation import Annotation

a = Annotation()  # Calling the class and assigning it to 'a' with the dataset as the argument
a.dataframe()
