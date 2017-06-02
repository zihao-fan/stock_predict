# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os
import json
from scipy import stats
from config import val_ratio, test_ratio
import cPickle
from __future__ import print_function

current_path = os.path.realpath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])

label_thresh = [-10.0, -0.03, 0.03, 10.0]

def csv_to_pd(filename):
    full_path = os.path.join(root_path, 'raw_data', filename)
    df = pd.read_csv(full_path)
    df.sort(columns=['secID'], inplace=True)
    df.set_index(keys=['secID'], drop=False, inplace=True)
    ids = df['secID'].unique().tolist()

    df.to_pickle(os.path.join(root_path, 'data', 'factor_pd.pickle'))
    with open (os.path.join(root_path, 'data', 'factor_ids.json'), 'w') as f:
        json.dump(ids, f)

    print 'Data saved.'

def read_price(filename):
    full_path = os.path.join(root_path, 'raw_data', filename)
    df = pd.read_csv(full_path, index_col=0).transpose()
    df.index = df.index.map(int)
    return df

def select_by_id(df, price, secid=''):
    if len(secid) > 0:
        part_df = df.loc[df.secID == secid].copy()
    else:
        part_df = df.copy()
    part_df.sort(columns=['date'], inplace=True)
    part_df = part_df.set_index(keys=['date'])
    part_df = part_df.drop(['secID'], axis=1)
    part_df = part_df.fillna(axis=1, method='ffill')
    part_df['close_price'] = pd.Series(0, index=part_df.index)
    
    current_price = price[secid]

    for index, row in part_df.iterrows():
        part_df['close_price'].loc[index] = current_price.loc[index]

    # add label
    part_df['next_month'] = (part_df['close_price'].shift(-1) - part_df['close_price']) / part_df['close_price']
    part_df['next_month'] = pd.cut(part_df['next_month'], bins=label_thresh, labels=[0,1,2]).astype(int)
    part_df = part_df[:-1]
    return part_df

def split_data(data):
    ntest = int(round(len(data) * (1 - test_ratio) ))
    nval = int(round(len(data[:ntest]) * (1 - val_ratio) ))

    nd_train, nd_val, nd_test = data[:nval], data[nval:ntest], data[ntest:]

    return nd_train, nd_val, nd_test

def rnn_data(data, time_steps):
    x_df = []
    y_df = []
    for i in range(len(data) - time_steps):
        # label
        y_df.append(data.iloc[i + time_steps, data.columns=='next_month'].as_matrix())
        # feature
        data_ = data.iloc[i: i + time_steps, data.columns!='next_month'].as_matrix()
        x_df.append(data_ if len(data_.shape) > 1 else [ [i] for i in data_ ])
    return np.asarray(x_df), np.asarray(y_df).squeeze()

def get_factor_dataset(time_steps):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    # read from local
    price = read_price('Daliy_ClosePrice.csv')
    ids = []
    with open(os.path.join(root_path, 'data', 'factor_ids.json'), 'r') as f:
        ids = json.load(f)
    df = pd.read_pickle(os.path.join(root_path, 'data', 'factor_pd.pickle'))
    for idx in ids:
        part_df = select_by_id(df, price, idx)
        if len(part_df) <= 1.5 * time_steps:
            continue
        # part_df['next_month'].describe()
        x_array, y_array = rnn_data(part_df, time_steps)
        # input
        train_xx, val_xx, test_xx = split_data(x_array)
        train_x.append(train_xx)
        val_x.append(val_xx)
        test_x.append(test_xx)

        # output
        train_yy, val_yy, test_yy = split_data(y_array)
        train_y.append(train_yy)
        val_y.append(val_yy)
        test_y.append(test_yy)
        print idx

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    val_x = np.concatenate(val_x, axis=0)
    val_y = np.concatenate(val_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    full_path = os.path.join(root_path, 'data', 'factor_dataset.cpickle')
    cPickle.dump((train_x, train_y, val_x, val_y, test_x, test_y), open(full_path, 'wb'))
    print 'Saved to', full_path

if __name__ == '__main__':
    # csv_to_pd('select_factor.csv')
    get_factor_dataset(24)