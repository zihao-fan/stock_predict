#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
import mxnet as mx

def mlp_sym(hidden_size, num_class, activation_func='tanh'):
    data = mx.sym.Variable('data')
    
    fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=hidden_size)
    act1 = mx.sym.Activation(data=fc1, name=activation_func+'1', act_type=activation_func)

    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=(hidden_size//2))
    act2 = mx.sym.Activation(data=fc2, name=activation_func+'2', act_type=activation_func)

    fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=num_class)
    mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    return mlp

if __name__ == '__main__':
    batch_size = 64
    shape = {'data':(batch_size, 7)}
    mx.viz.plot_network(symbol=mlp_sym(128, 2), shape=shape)