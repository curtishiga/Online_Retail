import pandas as pd
import numpy as np
import datetime

def split_date_monthday(X):
    # X is passed as a DataFrame or numpy array depending on config
    df_temp = pd.DataFrame(X)

    # Assuming single column input, get the first column and convert to datetime
    dt = pd.to_datetime(df_temp.iloc[:, 0])

    day = dt.dt.day.values
    month = dt.dt.month.values

    return np.column_stack([month, day])


def monthday_feature_names(transformer, input_features):
    return [f'{input_features[0]}_day', f'{input_features[0]}_month']


def split_date_month(X):
    # X is passed as a DataFrame or numpy array depending on config
    df_temp = pd.DataFrame(X)

    # Assuming single column input, get the first column and convert to datetime
    dt = pd.to_datetime(df_temp.iloc[:, 0])

    month = dt.dt.month.values

    return np.column_stack([month])


def month_feature_names(transformer, input_features):
    return [f'{input_features[0]}_month']


def split_date_day(X):
    # X is passed as a DataFrame or numpy array depending on config
    df_temp = pd.DataFrame(X)

    # Assuming single column input, get the first column and convert to datetime
    dt = pd.to_datetime(df_temp.iloc[:, 0])

    day = dt.dt.day.values

    return np.column_stack([day])


def day_feature_names(transformer, input_features):
    return [f'{input_features[0]}_day']
