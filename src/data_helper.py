# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import os
from os import listdir
from os.path import isfile, join
from config import INDEX_DIR, SH_DIR, SZ_DIR

index_list = ['开盘价', '最高价', '最低价', '收盘价', '后复权价', '前复权价', 
    '成交量', '成交额', '换手率', '流通市值', '总市值', 'MA_5', 'MA_10', 'MA_20', 'MA_30', 'MA_60',
    'MACD_DIF', 'MACD_DEA', 'MACD_MACD', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'rsi1', 'rsi2', 'rsi3', '振幅', '量比', '涨跌幅']
check_number = 10

def to_onehot(label_matrix, label_num):
    assert len(label_matrix.shape) == 1
    sample_num = label_matrix.shape[0]
    onehot = np.zeros((sample_num, label_num))
    for i in range(sample_num):
        onehot[i][label_matrix[i]] = 1
    return onehot

def time_to_onehot(y_matrix, label_num):
    assert len(y_matrix.shape) == 2
    sample_num = y_matrix.shape[0]
    timestep = y_matrix.shape[1]
    onehot = np.zeros((sample_num, timestep, label_num))
    for i in range(sample_num):
        for j in range(timestep):
            onehot[i, j, int(y_matrix[i,j])] = 1
    return onehot

def get_filenames_from_dir(my_dir, max_num=None):
    csv_files = [f for f in listdir(my_dir) if isfile(join(my_dir, f)) and f.endswith('.csv')]
    if max_num is not None:
        csv_files = csv_files[0:min(max_num, len(csv_files))]
        # csv_files = np.random.choice(csv_files, min(max_num, len(csv_files)), replace=False)
    return csv_files

def get_dataframe(stock_code, debug=False):
    if not stock_code.endswith('.csv'):
        filename = stock_code + '.csv'
    else:
        filename = stock_code
    if stock_code.startswith('sh'):
        file_path = os.path.join(SH_DIR, filename)
    else:
        file_path = os.path.join(SZ_DIR, filename)
    raw_df = pd.read_csv(file_path, index_col='交易日期').sort_index(axis=0, ascending=True)
    raw_df = raw_df.loc[:, index_list]
    if debug:
        return raw_df.head(check_number)
    else:
        return raw_df

def preprocess_dataset(train, val):
    mean_vec = np.mean(train, axis=0)
    max_vec = np.abs(np.max(train, axis=0)) + 1e-6
    train = (train - mean_vec) / max_vec
    val = (val - mean_vec) / max_vec
    return train, val

def get_mlp_dataset(stock_code, history_len=5, predict_offset=1, predict_stride=1):
    raw_df = get_dataframe(stock_code)
    matrix = raw_df.values[1:, 0:-1]
    labels = raw_df.values[1:, -1]

    data_point_number = ((matrix.shape[0] - history_len - predict_offset) // predict_stride) + 1
    if data_point_number <= 0:
        return (None, None), (None, None)
    val_size = data_point_number // 10
    train_size = data_point_number - val_size
    
    dataset_x = np.zeros((data_point_number, matrix.shape[1] * history_len), dtype='float32')
    dataset_label = np.zeros(data_point_number, dtype='int32')

    for i in range(data_point_number):
        idx = i * predict_stride
        dataset_x[i, :] = np.reshape(matrix[idx:idx+history_len, :], matrix.shape[1] * history_len)
        dataset_label[i] = 1 if labels[idx+history_len+predict_offset-1] > 0.0 else 0

    x_train = dataset_x[0:train_size, :]
    x_val = dataset_x[train_size:, :]
    label_train = dataset_label[0:train_size]
    label_val = dataset_label[train_size:]
    assert x_train.shape[0] + x_val.shape[0] == data_point_number
    x_train, x_val = preprocess_dataset(x_train, x_val)
    return (x_train, label_train), (x_val, label_val)

def get_rnn_dataset(stock_code, history_len=5, predict_offset=1, predict_stride=1):
    raw_df = get_dataframe(stock_code)
    matrix = raw_df.values[1:, 0:-1]
    labels = raw_df.values[1:, -1]

    data_point_number = ((matrix.shape[0] - history_len - predict_offset) // predict_stride) + 1
    if data_point_number <= 0:
        return (None, None, None), (None, None, None)
    val_size = data_point_number // 10
    train_size = data_point_number - val_size

    dataset_x = np.zeros((data_point_number, history_len, matrix.shape[1]), dtype='float32')
    dataset_y = np.zeros((data_point_number, history_len,), dtype='float32')
    dataset_label = np.zeros(data_point_number, dtype='int32')

    for i in range(data_point_number):
        idx = i * predict_stride
        dataset_x[i] = matrix[idx:idx+history_len, :]
        # dataset_y[i] = matrix[idx+1:idx+history_len+1, :]
        for j in range(history_len):
            dataset_y[i, j] = 1 if labels[idx+1+j] > 0.0 else 0
        dataset_label[i] = 1 if labels[idx+history_len+predict_offset-1] > 0.0 else 0

    x_train = dataset_x[0:train_size]
    x_val = dataset_x[-val_size:]
    y_train = dataset_y[0:train_size]
    y_val = dataset_y[-val_size:]
    label_train = dataset_label[0:train_size]
    label_val = dataset_label[-val_size:]

    x_train, x_val = preprocess_dataset(x_train, x_val)

    return (x_train, y_train, label_train), (x_val, y_val, label_val)

def get_dir_dataset(dir_name, data_set_type, max_num=None):
    assert data_set_type in ('rnn', 'mlp')
    csv_filenames = get_filenames_from_dir(dir_name, max_num)
    if data_set_type == 'rnn':
        x_train_all = []
        y_train_all = []
        label_train_all = []
        x_val_all = []
        y_val_all = []
        label_val_all = []
        for name in csv_filenames:
            (x_train, y_train, label_train), (x_val, y_val, label_val) = get_rnn_dataset(name)
            if x_train is None:
                continue
            x_train_all.append(x_train)
            y_train_all.append(y_train)
            label_train_all.append(label_train)
            x_val_all.append(x_val)
            y_val_all.append(y_val)
            label_val_all.append(label_val)
        return (np.concatenate(x_train_all), np.concatenate(y_train_all), np.concatenate(label_train_all)),\
            (np.concatenate(x_val_all), np.concatenate(y_val_all), np.concatenate(label_val_all))
    if data_set_type == 'mlp':
        x_train_all = []
        label_train_all = []
        x_val_all = []
        label_val_all = []
        for name in csv_filenames:
            (x_train, label_train), (x_val, label_val) = get_mlp_dataset(name)
            if x_train is None:
                continue
            x_train_all.append(x_train)
            label_train_all.append(label_train)
            x_val_all.append(x_val)
            label_val_all.append(label_val)
        return (np.concatenate(x_train_all), np.concatenate(label_train_all)),\
            (np.concatenate(x_val_all), np.concatenate(label_val_all))

if __name__ == '__main__':
    df = get_dataframe('sh600000')
    print(df.head())
    # (x_train, label_train), (x_val, label_val) = get_mlp_dataset('sh600000', predict_stride=2)
    # (x_train, y_train, label_train), (x_val, y_val, label_val) = get_rnn_dataset('sh600000')
    # print(x_train.shape, label_train.shape)