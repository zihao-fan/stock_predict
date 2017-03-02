#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
# import mxnet as mx

import data_helper
from data_helper import to_onehot, time_to_onehot
# import network_symbol
import network_model

# import logging
# logging.getLogger().setLevel(logging.DEBUG)

feature_dim = 27
batch_size = 16
epochs = 10
seq_len = 5

def mlp():
    hidden_size = 32
    # (x_train, label_train), (x_val, label_val) = data_helper.get_mlp_dataset('sh600000')
    (x_train, label_train), (x_val, label_val) = data_helper.get_dir_dataset(data_helper.SH_DIR, 'mlp')
    print('mlp train and val shape', x_train.shape, x_val.shape)
    label_train = to_onehot(label_train, 2)
    label_val = to_onehot(label_val, 2)
    print(x_train.shape, label_train.shape)
    model = network_model.mlp_model(feature_dim*seq_len, hidden_size, 2)
    model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    model.fit(x_train, label_train, batch_size=batch_size, nb_epoch=10, validation_data=(x_val, label_val))
    print('MLP model done.\n')

def lstm():
    hidden_size = 32
    # (x_train, y_train, label_train), (x_val, y_val, label_val) = data_helper.get_rnn_dataset('sh600000')
    (x_train, y_train, label_train), (x_val, y_val, label_val) = data_helper.get_dir_dataset(data_helper.SH_DIR, 'rnn')
    print('rnn train and val shape', x_train.shape, x_val.shape)
    label_train = to_onehot(label_train, 2)
    y_train = time_to_onehot(y_train, 2)
    label_val = to_onehot(label_val, 2)
    y_val = time_to_onehot(y_val, 2)
    model = network_model.lstm_model(feature_dim, seq_len, hidden_size, 2)
    # model.summary()
    model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(x_val, y_val))
    print('LSTM model done.')

if __name__ == '__main__':
    mlp()
    # lstm()