# -*- coding: utf-8 -*-
import numpy as np
import os
import network_model
import highfreq_helper
from highfreq_helper import root_path

batch_size = 32
epochs = 10

if __name__ == '__main__':
    datafile_path = os.path.join(root_path, 'data', '000905_20160506_20170506.data')
    data = highfreq_helper.read_pickle(datafile_path)

    time_steps = 10
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = highfreq_helper.get_rnn_dataset(data, time_steps)
    model = network_model.lstm_model(1, time_steps, 128, 1)
    model.fit(train_x, train_y, 
        batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))