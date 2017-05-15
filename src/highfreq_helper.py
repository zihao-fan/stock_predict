# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os
from scipy import stats
from keras.utils.np_utils import to_categorical

current_path = os.path.realpath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])

def read_pickle(file_path):
    return pd.read_pickle(file_path)

def split_data(data, val_ratio=0.1, test_ratio=0.2):
    ntest = int(round( len(data) * (1 - test_ratio) ))
    nval = int(round( len(data.iloc[:ntest]) * (1 - val_ratio) ))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test

def rnn_data(data, time_steps, skew, labels=False):
    rnn_df = []
    for i in range(len(data) - time_steps - skew):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps + skew].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps + skew])
        else:
            data_ = data.iloc[i + skew : i + skew + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [ [i] for i in data_ ])
    return np.asarray(rnn_df)

def get_rnn_predict_dataset(data, time_steps, skew):
    df_train, df_val, df_test = split_data(data['rate_cate'])
    df_train_y, df_val_y, df_test_y = split_data(data['rate_pred'])

    train_x = rnn_data(df_train, time_steps, 0, labels=False)
    train_y = rnn_data(df_train_y, time_steps, skew, labels=True)

    val_x = rnn_data(df_val, time_steps, 0, labels=False)
    val_y = rnn_data(df_val_y, time_steps, skew, labels=True)

    test_x = rnn_data(df_test, time_steps, 0, labels=False)
    test_y = rnn_data(df_test_y, time_steps, skew, labels=True)

    train_x, val_x, test_x = np.squeeze(train_x[:-skew]), np.squeeze(val_x[:-skew]), np.squeeze(test_x[:-skew])
    train_y, val_y, test_y = to_categorical(train_y), to_categorical(val_y), to_categorical(test_y)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def get_rnn_pretrain_dataset(data, time_steps, skew):
    df_train, df_val, df_test = split_data(data['rate_cate'])

    train_x = rnn_data(df_train, time_steps, 0, labels=False)
    train_y = rnn_data(df_train, time_steps, skew, labels=False)

    val_x = rnn_data(df_val, time_steps, 0, labels=False)
    val_y = rnn_data(df_val, time_steps, skew, labels=False)

    test_x = rnn_data(df_test, time_steps, 0, labels=False)
    test_y = rnn_data(df_test, time_steps, skew, labels=False)

    train_x, val_x, test_x = np.squeeze(train_x[:-skew]), np.squeeze(val_x[:-skew]), np.squeeze(test_x[:-skew])
    train_y, val_y, test_y = to_categorical(train_y), to_categorical(val_y), to_categorical(test_y)
    train_y = train_y.reshape(train_x.shape[0], train_x.shape[1], -1)
    val_y = val_y.reshape(val_x.shape[0], val_x.shape[1], -1)
    test_y = test_y.reshape(test_x.shape[0], test_x.shape[1], -1)
    
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

def get_equal_bin_edges(data, bin_num):
    ratio = np.ones(bin_num+1, dtype=np.float32)
    for i in range(bin_num):
        ratio[i] = ratio[i] * i / bin_num
    # print ratio
    bin_edges = stats.mstats.mquantiles(data, ratio)
    bin_edges[0] = -0.1
    bin_edges[bin_num] = 0.1
    return bin_edges

def add_rate(df):
    close_price = df['closePrice'].values
    rate_array = np.zeros_like(close_price)
    for i in range(rate_array.shape[0] - 1):
        rate_array[i + 1] = (close_price[i + 1] - close_price[i])/close_price[i]
    df = df.iloc[1:]
    rate_series = pd.Series(rate_array[1:], index=df.index)
    df['rate'] = rate_series
    return df

def add_rate_categories(df, binnum1, binnum2):
    rates = df['rate'].values
    bins = get_equal_bin_edges(df['rate'], binnum1)
    bins_pred = get_equal_bin_edges(df['rate'], binnum2)
    print 'bins', bins
    print 'bins pred', bins_pred
    categories = np.digitize(rates, bins)
    categories_pred = np.digitize(rates, bins_pred)
    df['rate_cate'] = pd.Series(categories, index=df.index)
    df['rate_pred'] = pd.Series(categories_pred, index=df.index)
    return df

if __name__ == '__main__':
    datafile_path = os.path.join(root_path, 'data', '000905_20100101_20170515.data')
    data = read_pickle(datafile_path)
    if 'rate' in data.columns:
        # print 'Adding rate to data'
        # data = add_rate(data)
        data = add_rate_categories(data, 100, 3)
        data.to_pickle(datafile_path)
        print 'New dataframe save to', datafile_path
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_rnn_pretrain_dataset(data, 20, 1)
    # (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_rnn_predict_dataset(data, 20, 20)
    print train_x[0], train_y[0]
    print train_x.shape, train_y.shape