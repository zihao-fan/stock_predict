# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os
from scipy import stats
from keras.utils.np_utils import to_categorical
from config import INPUT_SKEW, INPUT_BIN
from config import OUTPUT_SKEW, OUTPUT_BIN
from config import val_ratio, test_ratio
from config import time_steps
from sklearn.preprocessing import StandardScaler
from __future__ import print_function

current_path = os.path.realpath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])

def read_pickle(file_path):
    return pd.read_pickle(file_path)

def split_data(data):
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
    df_train, df_val, df_test = split_data(data[['rate_cate', 'volume_in', 'rate_pred']])
    # df_train_y, df_val_y, df_test_y = split_data(data['rate_pred'])

    train_x = rnn_data(df_train['rate_cate'], time_steps, 0, labels=False)
    train_vol = rnn_data(df_train['volume_in'], time_steps, 0, labels=False)
    train_y = rnn_data(df_train['rate_pred'], time_steps, skew, labels=True)

    val_x = rnn_data(df_val['rate_cate'], time_steps, 0, labels=False)
    val_vol = rnn_data(df_val['volume_in'], time_steps, 0, labels=False)
    val_y = rnn_data(df_val['rate_pred'], time_steps, skew, labels=True)

    test_x = rnn_data(df_test['rate_cate'], time_steps, 0, labels=False)
    test_vol = rnn_data(df_test['volume_in'], time_steps, 0, labels=False)
    test_y = rnn_data(df_test['rate_pred'], time_steps, skew, labels=True)

    train_x, val_x, test_x = np.squeeze(train_x[:-skew]), np.squeeze(val_x[:-skew]), np.squeeze(test_x[:-skew])
    train_vol, val_vol, test_vol = np.squeeze(train_vol[:-skew]), np.squeeze(val_vol[:-skew]), np.squeeze(test_vol[:-skew])
    train_y, val_y, test_y = to_categorical(train_y), to_categorical(val_y), to_categorical(test_y)

    return (train_x, train_vol, train_y), (val_x, val_vol, val_y), (test_x, test_vol, test_y)


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
    bin_edges[0] = -100.0
    bin_edges[bin_num] = 100.0
    return bin_edges

def add_rate(df, skew1, skew2):
    close_price = df['closePrice'].values
    total_volume = df['totalVolume'].values
    rate_array_1 = np.zeros_like(close_price)
    rate_array_2 = np.zeros_like(close_price)
    volume_array_1 = np.zeros_like(total_volume)
    for i in range(rate_array_1.shape[0] - skew1):
        # rate_array_1[i + skew1] = (close_price[i + skew1] - close_price[i]) / close_price[i] # rate return
        rate_array_1[i + skew1] = np.log(close_price[i + skew1]) - np.log(close_price[i])
        volume_array_1[i + skew1] = np.log(total_volume[i + skew1] + 1e-6) - np.log(total_volume[i] + 1e-6)
    for i in range(rate_array_2.shape[0] - skew2):
        # rate_array_2[i + skew2] = (close_price[i + skew2] - close_price[i]) / close_price[i] # rate return
        rate_array_2[i + skew2] = np.log(close_price[i + skew2]) - np.log(close_price[i])
    df = df.iloc[max(skew1, skew2):]
    rate_series_1 = pd.Series(rate_array_1[max(skew1, skew2):], index=df.index)
    rate_series_2 = pd.Series(rate_array_2[max(skew1, skew2):], index=df.index)
    volume_series_1 = pd.Series(volume_array_1[max(skew1, skew2):], index=df.index)
    df['rate_'+str(skew1)] = rate_series_1
    df['rate_'+str(skew2)] = rate_series_2
    df['volume_'+str(skew1)] = volume_series_1
    return df

def add_rate_categories(df, skew1, skew2, binnum1, binnum2):
    rates_1 = df['rate_'+str(skew1)].values
    train_1, _, _ = split_data(df['rate_'+str(skew1)])
    
    volume_1 = df['volume_'+str(skew1)].values
    train_volume, _, _ = split_data(df['volume_'+str(skew1)])
    
    rates_2 = df['rate_'+str(skew2)].values
    train_2, _, _ = split_data(df['rate_'+str(skew2)])
    
    # bins = get_equal_bin_edges(train_1.values, binnum1)
    # bins_volume = get_equal_bin_edges(train_volume.values, binnum1)
    # bins_pred = get_equal_bin_edges(train_2.values, binnum2)

    bins = get_equal_bin_edges(df['rate_'+str(skew1)].values, binnum1)
    bins_volume = get_equal_bin_edges(df['volume_'+str(skew1)].values, binnum1)
    bins_pred = get_equal_bin_edges(df['rate_'+str(skew2)].values, binnum2)
    
    print('bins', bins)
    print('bins volume', bins_volume)
    print('bins pred', bins_pred)
    categories = np.digitize(rates_1, bins)
    categories_volume = np.digitize(volume_1, bins_volume)
    categories_pred = np.digitize(rates_2, bins_pred)
    df['rate_cate'] = pd.Series(categories, index=df.index)
    df['volume_in'] = pd.Series(categories_volume, index=df.index)
    df['rate_pred'] = pd.Series(categories_pred, index=df.index)
    return df

def add_label_df(df):
    df['next_min'] = df['closePrice'].shift(-1)
    df['next_hour'] = df['closePrice'].shift(-60)
    df['next_day'] = df['closePrice'].shift(-240)

    ascent = lambda row, target: 1 if row['closePrice'] <= row[target] else 0
    df['ascent_next'] = df.apply(ascent, axis=1, args=['next_min']).astype(int)
    df['ascent_hour'] = df.apply(ascent, axis=1, args=['next_hour']).astype(int)
    df['ascent_day'] = df.apply(ascent, axis=1, args=['next_day']).astype(int)

    return df 

def make_data(data, time_steps, label):
    serX, serY = [], []
    ydata = data[label].as_matrix()
    data = data.drop([label], axis=1).as_matrix()
    # data = data.as_matrix()
    for i in xrange(len(data)-time_steps):
        serX.append(data[i:i+time_steps])
        serY.append(ydata[i+time_steps])
    inputX = np.array(serX)
    inputY = np.array(serY)
    return inputX, inputY

def normalize_dataset(train, val, test):
    mean_vec = np.mean(train, axis=0)
    std_vec = np.std(train, axis=0)
    train = (train - mean_vec) / std_vec
    val = (val - mean_vec) / std_vec
    test = (test - mean_vec) / std_vec
    return train, val, test

def generate_rnn_data(df, time_steps, label):
    '''
    
    drop the data

    '''
    df = df.drop(['currencyCD'], axis=1)

    df['null_count'] = df.apply(lambda row: row.isnull().sum(), axis=1)
    df = df[df['null_count']==0]
    df = df.drop(['null_count'], axis=1)

    to_drop = ['unit', 'ticker', 'exchangeCD', 'shortNM', 'barTime']
    to_drop_start = ['ascent', 'next']
    for col in df.columns:
        for x in to_drop_start:
            if col.startswith(x):
                to_drop.append(col)
                break
    if label in to_drop:
        to_drop.remove(label)
    df = df.drop(to_drop, axis=1)

    '''
    split the data
    '''

    df_train, df_val, df_test = split_data(df)

    '''
    make it input and output
    '''
    train_x, train_y = make_data(df_train, time_steps, label)
    val_x, val_y = make_data(df_val, time_steps, label)
    test_x, test_y = make_data(df_test, time_steps, label)

    train_x, val_x, test_x = normalize_dataset(train_x, val_x, test_x)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


if __name__ == '__main__':
    rawdata_path = os.path.join(root_path, 'raw_data', '000905_20100101_20170525.data')
    outpath = os.path.join(root_path, 'data', '000905_20100101_20170525.data')
    rawdata = read_pickle(rawdata_path)
    print(rawdata.columns)
    # df = rawdata.iloc[:]
    # df = add_label_df(df)
    # df.to_pickle(outpath)
    # print 'New dataframe save to', outpath
    # df = read_pickle(outpath)
    # (train_x, train_y), (val_x, val_y), (test_x, test_y) = generate_rnn_data(df, 180, 'ascent_hour')
    # print train_x.shape, train_y.shape

    ''' ========================================================= '''
    print('Adding rate to data')
    data = add_rate(rawdata, INPUT_SKEW, OUTPUT_SKEW)
    data = add_rate_categories(data, INPUT_SKEW, OUTPUT_SKEW, INPUT_BIN, OUTPUT_BIN)
    print(data.describe())
    data.to_pickle(outpath)
    print('New dataframe save to', outpath)
    (train_x, train_vol, train_y), (val_x, val_vol, val_y), (test_x, test_vol, test_y) = get_rnn_predict_dataset(data, 20, OUTPUT_SKEW)
    print(train_x[0], train_vol[0], train_y[0])
    print(train_x.shape, train_vol.shape, train_y.shape)