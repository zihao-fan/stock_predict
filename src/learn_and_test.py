# -*- coding: utf-8 -*-

import numpy as np
import network_model
import data_helper
from data_helper import to_onehot, time_to_onehot
from data_helper import index_list


feature_dim = len(index_list) - 1
batch_size = 32
epochs = 10
seq_len = 5
num_class = 3

def mlp(stock_list):
    hidden_size = 16
    for stock_name in stock_list:
        (x_train, label_train), (x_val, label_val) = data_helper.get_mlp_dataset(stock_name, label='rise/fall')
        print('mlp train and val shape', x_train.shape, x_val.shape)
        label_train = to_onehot(label_train, num_class)
        label_val = to_onehot(label_val, num_class)
        print(x_train.shape, label_train.shape)
        model = network_model.mlp_model(feature_dim, seq_len, hidden_size, num_class)
        model.fit(x_train, label_train, batch_size=batch_size, epochs=10, validation_data=(x_val, label_val))
        scores = model.evaluate(x_val, label_val, batch_size=x_val.shape[0])
        print('\n', model.metrics_names[1], scores[1] * 100)
    print('MLP model done.\n')

def lstm(stock_list):
    hidden_size = 16
    for stock_name in stock_list:
        (x_train, y_train, label_train), (x_val, y_val, label_val) = data_helper.get_rnn_dataset(stock_name, label='rise/fall')
        print('rnn train and val shape', x_train.shape, x_val.shape)
        label_train = to_onehot(label_train, num_class)
        y_train = time_to_onehot(y_train, num_class)
        label_val = to_onehot(label_val, num_class)
        y_val = time_to_onehot(y_val, num_class)
        model = network_model.lstm_model(feature_dim, seq_len, hidden_size, num_class)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_val, y_val))
        scores = model.evaluate(x_val, y_val, batch_size=x_val.shape[0])
        print('\n', model.metrics_names[1], scores[1] * 100)
    print('LSTM model done.')

if __name__ == '__main__':
    mlp(['sh600000', 'sh600004', 'sh600005'])
    lstm(['sh600004', 'sh600005', 'sh600006'])