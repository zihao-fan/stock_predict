# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os
import visualization

current_path = os.path.realpath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])

def read_pickle(file_path):
    return pd.read_pickle(file_path)

def split_data(data, val_ratio=0.1, test_ratio=0.2):
    ntest = int(round( len(data) * (1 - test_ratio) ))
    nval = int(round( len(data.iloc[:ntest]) * (1 - val_ratio) ))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test

def rnn_data(data, time_steps, labels=False):
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i : i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [ [i] for i in data_ ])
    return np.asarray(rnn_df)

def get_rnn_dataset(data, time_steps):
    df_train, df_val, df_test = split_data(data['closePrice'])

    train_x = rnn_data(df_train, time_steps, labels=False)
    train_y = rnn_data(df_train, time_steps, labels=True)

    val_x = rnn_data(df_val, time_steps, labels=False)
    val_y = rnn_data(df_val, time_steps, labels=True)

    test_x = rnn_data(df_test, time_steps, labels=False)
    test_y = rnn_data(df_test, time_steps, labels=True)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

if __name__ == '__main__':
    datafile_path = os.path.join(root_path, 'data', '000905_20160506_20170506.data')
    data = read_pickle(datafile_path)

    df_train, df_val, df_test = split_data(data['closePrice'])
    
    df_train = df_train.iloc[0:30]
    print df_train
    time_steps = 10
    train_x = rnn_data(df_train, time_steps, labels=False)
    train_y = rnn_data(df_train, time_steps, labels=True)
    print train_x.shape, train_y.shape