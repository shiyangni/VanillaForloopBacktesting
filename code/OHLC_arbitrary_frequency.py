import pandas as pd
import numpy as np


#################### Main API ########################
def add_time(df):
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def get_OHLC_with_freq(df, freq, open_name='open', close_name='close', high_name='high', low_name='low', time_name='time'):
    ### Converts one OHLC data frame to OHLC with lower frequencies.
    ### Preferrably the df contains 'volume' and 'count' columns.
    if ('volume' in df.columns) and ('count' in df.columns):
        result = df.groupby(pd.Grouper(key=time_name, freq=freq)).agg({'volume': 'sum', 'count': 'sum'})
    elif ('volume' in  df.columns):
        result = df.groupby(pd.Grouper(key=time_name, freq=freq)).agg({'volume': 'sum'})
    else:
        print('The dataframe has neither volume or count.')
        result = pd.DataFrame(index = df.groupby(pd.Grouper(key=time_name, freq=freq)).agg({'open': 'count'}).index)
    result[open_name] = df.groupby(pd.Grouper(key=time_name, freq=freq)).apply(get_open, open_name=open_name)
    result[close_name] = df.groupby(pd.Grouper(key=time_name, freq=freq)).apply(get_close, close_name=close_name)
    result[high_name] = df.groupby(pd.Grouper(key=time_name, freq=freq)).apply(get_high, high_name=high_name)
    result[low_name] = df.groupby(pd.Grouper(key=time_name, freq=freq)).apply(get_low, low_name=low_name)
    return result.reset_index()

#################### Below are utility functions ###################

def get_open(df, open_name='open'):
    try:
        opening = df[open_name].values[0]
    except IndexError:
        opening = None
    return opening

def get_close(df, close_name='close'):
    try:
        closing = df[close_name].values[df.shape[0] - 1]
    except IndexError:
        closing = None
    return closing

def get_high(df, high_name='high'):
    try:
        high = df[high_name].max()
    except IndexError:
        high = None
    return high

def get_low(df, low_name='low'):
    try:
        low = df[low_name].min()
    except IndexError:
        low = None
    return low
