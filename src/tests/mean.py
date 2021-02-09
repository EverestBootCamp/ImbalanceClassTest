
def aggregate_mean(df, column):
    return df.groupby("class")[column].mean().to_dict()

