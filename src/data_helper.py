# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import os
from os import listdir
from os.path import isfile, join
from config import INDEX_DIR, SH_DIR, SZ_DIR

# index_list = ['开盘价', '最高价', '最低价', '收盘价', '后复权价', '前复权价', 
#     '成交量', '成交额', '换手率', '流通市值', '总市值', 'MA_5', 'MA_10', 'MA_20', 'MA_30', 'MA_60',
#     'MACD_DIF', 'MACD_DEA', 'MACD_MACD', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'rsi1', 'rsi2', 'rsi3', '振幅', '量比', '涨跌幅']
index_list = ['前复权价', '换手率', 'MACD_MACD', '振幅', '量比', '涨跌幅']
# index_list = ['前复权价', 'MACD_MACD', '涨跌幅']
check_number = 10
days_skipped = 365
eps = 0.01
threshold_bins = [-1., -2*eps, 2*eps, 1.]
threshold_labels = [0, 1, 2]

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

def distribution_in_labels(label_input):
    label = label_input.flatten()
    value_dict = {}
    total_num = label.shape[0]
    for i in range(total_num):
        current = label[i]
        if current not in value_dict:
            value_dict[current] = 1
        else:
            value_dict[current] += 1
    for key, value in value_dict.items():
        value_dict[key] = float(value) / total_num
    return value_dict

def get_market_label(df, look_back=3):
    '''
    TODO
    '''
    labels = df.values[:, -1]
    num = labels.shape[0]
    new_labels = np.zeros(num, dtype=np.int32)
    for i in range(num):
        if i < look_back:
            continue
        rise = True
        fall = True
        for j in range(look_back):
            rise = rise and labels[i - j] > 0.
            fall = fall and labels[i - j] < 0.
        if rise or fall:
            new_labels[i] = 1
    counter = 0
    for index, row in df.iterrows():
        df.set_value(index, '涨跌幅', new_labels[counter])
        counter += 1
    return df

def dataset_sampling(x, y, ratio=0.2):
    number = x.shape[0]
    selected = np.random.choice(number, int(ratio*number), replace=False)
    return x[selected], y[selected]

def get_mlp_dataset(stock_code, history_len=5, predict_offset=1, predict_stride=1, label='rise/fall'):
    assert label in ['rise/fall', 'market'], 'Please give supported label type.'
    raw_df = get_dataframe(stock_code)
    raw_df = raw_df[days_skipped:] # skip the first year for stable data
    # print(raw_df['涨跌幅'].describe())
    if label == 'rise/fall':
        raw_df['涨跌幅'] = pd.cut(raw_df['涨跌幅'], threshold_bins, labels=threshold_labels) # binning the label
    elif label == 'market':
        raw_df = get_market_label(raw_df)
    # print('The rise/fall distribution in', stock_code, 'is\n', raw_df['涨跌幅'].value_counts(normalize=True))
    matrix = raw_df.values[:, 0:-1]
    labels = raw_df.values[:, -1]
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
        dataset_label[i] = labels[idx+history_len+predict_offset-1]

    x_train = dataset_x[0:train_size, :]
    x_val = dataset_x[train_size:, :]
    label_train = dataset_label[0:train_size]
    label_val = dataset_label[train_size:]
    assert x_train.shape[0] + x_val.shape[0] == data_point_number

    train_dist = distribution_in_labels(label_train)
    val_dist = distribution_in_labels(label_val)
    print('Train label distribution', train_dist, 'Val label distribution', val_dist)
    # x_train, x_val = preprocess_dataset(x_train, x_val)
    return (x_train, label_train), (x_val, label_val)

def get_rnn_dataset(stock_code, history_len=5, predict_offset=1, predict_stride=1, label='rise/fall'):
    assert label in ['rise/fall', 'market'], 'Please give supported label type.'
    raw_df = get_dataframe(stock_code)
    raw_df = raw_df[days_skipped:]
    if label == 'rise/fall':
        raw_df['涨跌幅'] = pd.cut(raw_df['涨跌幅'], threshold_bins, labels=threshold_labels) # binning the label
    elif label == 'market':
        raw_df = get_market_label(raw_df)
    # print('The rise/fall distribution in', stock_code, 'is\n', raw_df['涨跌幅'].value_counts(normalize=True))
    matrix = raw_df.values[:, 0:-1]
    labels = raw_df.values[:, -1]

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
        for j in range(history_len):
            dataset_y[i, j] = labels[idx + 1 + j]
        dataset_label[i] = labels[idx + history_len + predict_offset - 1]

    x_train = dataset_x[0:train_size]
    x_val = dataset_x[-val_size:]
    y_train = dataset_y[0:train_size]
    y_val = dataset_y[-val_size:]
    label_train = dataset_label[0:train_size]
    label_val = dataset_label[-val_size:]

    train_dist = distribution_in_labels(y_train)
    val_dist = distribution_in_labels(y_val)
    print('Train y distribution', train_dist, 'Val y distribution', val_dist)

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
    (x_train, label_train), (x_val, label_val) = get_mlp_dataset('sh600000', label='market')