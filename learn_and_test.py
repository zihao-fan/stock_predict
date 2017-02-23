#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
import mxnet as mx

import data_helper
import network_symbol

import logging
logging.getLogger().setLevel(logging.DEBUG)

def mlp():
    batch_size = 16
    epochs = 10
    
    (x_train, label_train), (x_val, label_val) = data_helper.get_mlp_dataset('sh600000')
    train_iter = mx.io.NDArrayIter(x_train, label_train, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(x_val, label_val, batch_size)
    
    mlp = network_symbol.mlp_sym(128, 2)
    adam = mx.optimizer.Adam()
    model = mx.mod.Module(symbol=mlp, context=mx.gpu(3))

    model.fit(
        train_iter,
        eval_data=val_iter,
        batch_end_callback = mx.callback.Speedometer(batch_size, 100),
        optimizer='adam',
        num_epoch=epochs
    )

def build_dict(my_input):
    time_steps = my_input.shape[1]
    my_dict = {}
    for i in range(time_steps):
        my_dict['data/' + str(i)] = my_input[:, i, :].squeeze()
    return my_dict

def lstm():
    batch_size = 16
    epochs = 10
    num_lstm_layer = 2
    seq_len = 5
    input_dim = 27
    num_hidden = 32

    (x_train, y_train, label_train), (x_val, label_val) = data_helper.get_rnn_dataset('sh600000')

    x_train = build_dict(x_train)
    x_val = build_dict(x_val)

    train_iter = mx.io.NDArrayIter(x_train, label_train, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(x_val, label_val, batch_size)

    print(train_iter.provide_data, train_iter.provide_label)

    my_lstm = network_symbol.lstm_unroll(
        num_lstm_layer, 
        seq_len,
        input_dim,
        num_hidden=num_hidden,
        num_embed=None,
        num_label=2, 
        dropout=0.2)

    model = mx.mod.Module(symbol=my_lstm, context=mx.gpu(3))
    model.bind(data_shapes=train_iter.provide_data,
         label_shapes=train_iter.provide_label)
    # model.fit(
    #     train_iter,
    #     eval_data=val_iter,
    #     batch_end_callback = mx.callback.Speedometer(batch_size, 100),
    #     optimizer='adam',
    #     num_epoch=epochs,
    #     initializer=mx.init.Xavier(factor_type="in", magnitude=2.34)
    # )

if __name__ == '__main__':
    lstm()