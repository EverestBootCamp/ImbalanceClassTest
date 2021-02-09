import pandas as pd

def mean(df, column):
    return df.groupby("class")[column].mean().to_dict()

def pipeline(column):
    data = pd.read_csv("./data_transformed.csv")
    data = aggregate_mean(data, column)
    return data