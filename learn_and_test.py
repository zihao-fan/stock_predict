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

def main():
    batch_size = 64
    epochs = 10
    
    (x_train, label_train), (x_val, label_val) = data_helper.get_mlp_dataset('sh600000')
    train_iter = mx.io.NDArrayIter(x_train, label_train, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(x_val, label_val, batch_size)
    
    mlp = network_symbol.mlp_sym(128, 2)
    adam = mx.optimizer.Adam()
    model = mx.model.FeedForward(
        symbol = mlp,
        num_epoch = epochs,
        optimizer = adam,
        ctx=mx.gpu(3)
    )

    model.fit(
        X=train_iter,
        eval_data=val_iter,
        batch_end_callback = mx.callback.Speedometer(batch_size, 1)
    )

if __name__ == '__main__':
    main()