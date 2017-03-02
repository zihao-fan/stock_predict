#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import math
import os
from os import listdir
from os.path import isfile, join

INDEX_DIR = '/research/zihao/stock_dataset/index'
SH_DIR = '/research/zihao/stock_dataset/sh'
SZ_DIR = '/research/zihao/stock_dataset/sz'
index_list = ['开盘价', '最高价', '最低价', '收盘价', '后复权价', '前复权价', 
    '成交量', '成交额', '换手率', '流通市值', '总市值', 'MA_5', 'MA_10', 'MA_20', 'MA_30', 'MA_60',
    'MACD_DIF', 'MACD_DEA', 'MACD_MACD', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'rsi1', 'rsi2', 'rsi3', '振幅', '量比', '涨跌幅']
# index_list = index_list[-3:]
check_number = 10

def to_onehot(label_matrix, label_num):
    assert len(label_matrix.shape) == 1
    sample_num = label_matrix.shape[0]
    onehot = np.zeros((sample_num, label_num))
    for i in range(sample_num):
        onehot[i][label_matrix[i]] = 1
    return onehot

def get_filenames_from_dir(my_dir):
    csv_files = [f for f in listdir(my_dir) if isfile(join(my_dir, f))]
    return csv_files

def get_dataframe(stock_code, debug=False):
    filename = stock_code + '.csv'
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
    std_vec = np.std(train, axis=0)
    train = (train - mean_vec) / std_vec
    val = (val - mean_vec) / std_vec
    return train, val

def get_mlp_dataset(stock_code, history_len=5, predict_offset=1, predict_stride=1):
    raw_df = get_dataframe(stock_code)
    matrix = raw_df.values[1:, 0:-1]
    labels = raw_df.values[1:, -1]

    data_point_number = ((matrix.shape[0] - history_len - predict_offset) // predict_stride) + 1
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
    # mean_vec = np.min(x_train, axis=0)
    # for i in range(len(index_list)):
    #     if math.isnan(mean_vec[i]):
    #         print(index_list[i])
    return (x_train, label_train), (x_val, label_val)

def get_rnn_dataset(stock_code, history_len=5, predict_offset=1, predict_stride=1):
    raw_df = get_dataframe(stock_code)
    matrix = raw_df.values[1:, 0:-1]
    labels = raw_df.values[1:, -1]

    data_point_number = ((matrix.shape[0] - history_len - predict_offset) // predict_stride) + 1
    val_size = data_point_number // 10
    train_size = data_point_number - val_size

    dataset_x = np.zeros((data_point_number, history_len, matrix.shape[1]), dtype='float32')
    dataset_y = np.zeros((data_point_number, history_len, matrix.shape[1]), dtype='float32')
    dataset_label = np.zeros(data_point_number, dtype='int32')

    for i in range(data_point_number):
        idx = i * predict_stride
        dataset_x[i] = matrix[idx:idx+history_len, :]
        dataset_y[i] = matrix[idx+1:idx+history_len+1, :]
        dataset_label[i] = 1 if labels[idx+history_len+predict_offset-1] > 0.0 else 0

    x_train = dataset_x[0:train_size]
    x_val = dataset_x[-val_size:]
    y_train = dataset_y[0:train_size]
    y_val = dataset_x[-val_size:]
    label_train = dataset_label[0:train_size]
    label_val = dataset_label[-val_size:]

    x_train, x_val = preprocess_dataset(x_train, x_val)
    # print(np.mean(x_train, axis=0), np.mean(x_val, axis=0))
    # y_train, y_val = preprocess_dataset(y_train, y_val)

    return (x_train, y_train, label_train), (x_val, label_val)

def main():
    sh_filenames = get_filenames_from_dir(SH_DIR)

if __name__ == '__main__':
    # (x_train, label_train), (x_val, label_val) = get_mlp_dataset('sh600000', predict_stride=2)
    (x_train, y_train, label_train), (x_val, label_val) = get_rnn_dataset('sh600000')
    print(x_train.shape, label_train.shape)