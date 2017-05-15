# -*- coding: utf-8 -*-
import numpy as np
import os
import network_model
import highfreq_helper
from highfreq_helper import root_path

batch_size = 32
epochs = 20
time_steps = 20
embedding_dim = 100
hidden_size = 100
bins_num = 100

models_path = os.path.join(root_path, 'models')

if __name__ == '__main__':
    datafile_path = os.path.join(root_path, 'data', '000905_20100101_20170515.data')
    data = highfreq_helper.read_pickle(datafile_path)

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = highfreq_helper.get_rnn_pretrain_dataset(data, 
                                                                            time_steps, 1)
    model = network_model.lstm_pretrain_model(bins_num, 
                                            time_steps, 
                                            embedding_dim, 
                                            hidden_size)
    model_name = 'pretrain.model'
    for e in range(epochs):
        print '------------Pretraining model-----------, epoch', e
        shuffled_rank = np.random.permutation(train_x.shape[0])
        train_x = train_x[shuffled_rank]
        train_y = train_y[shuffled_rank]
        model.fit(train_x, train_y, batch_size=batch_size, validation_data=(val_x, val_y))
        model.save(os.path.join(models_path, model_name))
        print 'model saved to', os.path.join(models_path, model_name)