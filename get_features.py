import time
import json
from kafka import KafkaConsumer
from kafka import KafkaProducer
import numpy as np
import pandas as pd

def get_dayofweek(df):
    possible_categories = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dtype = pd.CategoricalDtype(categories=possible_categories)
    cat = pd.Series(df['Date'].dt.day_name(), dtype=dtype)
    df = pd.concat((df, pd.get_dummies(cat).astype(float)), axis=1)
    return df

def get_month(df):
    possible_categories = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    dtype = pd.CategoricalDtype(categories=possible_categories)
    cat = pd.Series(df['Date'].dt.month_name(), dtype=dtype)
    df = pd.concat((df, pd.get_dummies(cat).astype(float)), axis=1)
    return df

def get_time_features_from_dataframe(df):
    df = get_dayofweek(df)
    df = get_month(df)
    return df

def get_z_score_for_dataframe(df, column, window_size=100):
    def zscore_func(x):
        return (x[-1] - x[:-1].mean())/x[:-1].std(ddof=0)
    df['Zscore'] = df[column].rolling(window=window_size+1).apply(zscore_func).fillna(0)
    return df

def get_mov_avg_for_dataframe(df, column, window_size=100):
    def moving_avg(x):
        return x[:-1].mean()
    for i in np.logspace(0,np.log(window_size)/np.log(10),5):
        df['Mean_'+str(int(i))] = df[column].rolling(window=int(i)).apply(moving_avg).fillna(0)
    return df

def get_roc_for_dataframe(df, column, window_size=100):
    for i in np.logspace(0,np.log(window_size)/np.log(10),5):
        df['ROC_'+str(int(i))] = df[column].pct_change(periods=int(i)).fillna(0)
    return df


class Feature_real_time():
    def __init__(self, tickers, useful_index="Close", window_size=100):
        self.useful_index = useful_index
        self.window_size = window_size
        self.history = {}
        for ticker in tickers:
            self.history[ticker] = []


    def update_history(self, element, key):
        if len(self.history[key]) == self.window_size:
            self.history[key].pop(0)
        self.history[key].append(element)

    def get_zscore(self, element, key):
        if len(self.history[key]) >= self.window_size and np.std(np.array(self.history[key])) != 0:
            return (element - np.mean(np.array(self.history[key])))/np.std(np.array(self.history[key]))
        else:
            return 0

    def get_mov_avg(self, key, window_size):
        if len(self.history[key]) >= window_size:
            array_corp = np.array(self.history[key])[-window_size:]
            return np.mean(array_corp)
        else:
            return 0

    def get_roc(self, element, key, window_size):
        if len(self.history[key]) >= window_size:
            return element/np.array(self.history[key])[-window_size] - 1
        else:
            return 0

    def get_feature_real_time_stream(self, stream):
        input_featured = {}
        for key in stream.keys():
            data = stream[key]
            self.update_history(data[self.useful_index], key)
            data["Zscore"] = self.get_zscore(data[self.useful_index], key)
            for i in np.logspace(0,np.log(self.window_size)/np.log(10),1):
                data['Mean_'+str(int(i))] = self.get_mov_avg( key, int(i))
            for i in np.logspace(0,np.log(self.window_size)/np.log(10),1):
                data['ROC_'+str(int(i))] = self.get_roc(data[self.useful_index], key, int(i))
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S.%f')
            df_data = pd.DataFrame(data, index = [0])
            df_data = get_time_features_from_dataframe(df_data)
            df_data = df_data.drop('Date', axis=1)
            df_data[['Open', 'High', 'Low', 'Close']] = df_data[['Open', 'High', 'Low', 'Close']].apply(np.log)
            df_data['Returns'] = df_data['Close']/df_data['Open'] - 1

            input_featured[key] = df_data.loc[0].to_dict()

        return input_featured

def get_features_for_past_data(data, useful_index="Close", window_size=100):

    for key in data.keys():
        df_data = data[key]
        df_data['Date'] = df_data.index
        df_data = df_data.drop('Volume', axis=1)
        df_data = df_data.drop('Dividends', axis=1)
        df_data = df_data.drop('Stock Splits', axis=1)
        df_data['Returns'] = (df_data['Close']/df_data['Open'] - 1)
        # df_data = get_time_features_from_dataframe(df_data)
        df_data = get_mov_avg_for_dataframe(df_data,column=useful_index, window_size=window_size)
        df_data = get_z_score_for_dataframe(df_data,column=useful_index, window_size=window_size)
        df_data = get_roc_for_dataframe(df_data,column=useful_index, window_size=window_size)
        df_data[['Open', 'High', 'Low', 'Close']] = df_data[['Open', 'High', 'Low', 'Close']].apply(np.log)
        df_data = df_data.drop('Date', axis=1)
        data[key] = df_data
    
    return data

