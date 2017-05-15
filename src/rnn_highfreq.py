# -*- coding: utf-8 -*-
import numpy as np
import os
import network_model
import highfreq_helper
from highfreq_helper import root_path
from keras.models import load_model

batch_size = 32
epochs = 10
time_steps = 20
embedding_dim = 100
hidden_size = 100
bins_num = 100
pretrain_skew = 1
predict_skew = 20
predict_class = 3

models_path = os.path.join(root_path, 'models')
results_path = os.path.join(root_path, 'results')

datafile_path = os.path.join(root_path, 'data', '000905_20100101_20170515.data')
data = highfreq_helper.read_pickle(datafile_path)

def evaluate(prediction, label):
    total_number = 0
    hit_number = 0
    rise_fall_total = 0
    rise_fall_hit = 0
    for i in range(len(prediction)):
        # overall result
        total_number += 1
        if prediction[i] == label[i]:
            hit_number += 1
        
        # rise fall result
        if label[i] == 1 or label[i] == 3:
            rise_fall_total += 1
            if prediction[i] == label[i]:
                rise_fall_hit += 1
        
    return float(hit_number) / total_number, float(rise_fall_hit) / rise_fall_total

def get_stats_dict(array):
    stats = {}
    for i in range(len(array)):
        if array[i] not in stats:
            stats[array[i]] = 0
        else:
            stats[array[i]] += 1
    for key, value in stats.items():
        stats[key] = float(value) / len(array)
    return stats

def test(model_name):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = highfreq_helper.get_rnn_predict_dataset(data, time_steps, predict_skew)
    model = load_model(os.path.join(models_path, model_name))
    print 'Testing the model'
    prediction = model.predict(test_x)
    labels = np.argmax(test_y, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    
    prediction_stats = get_stats_dict(prediction)
    label_stats = get_stats_dict(labels)
    print 'prediciton stats', prediction_stats
    print 'label stats', label_stats

    acc, accrf = evaluate(prediction, labels)
    print 'acc', acc, 'acc rise fall', accrf

def pretrain():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = highfreq_helper.get_rnn_pretrain_dataset(data, 
                                                                        time_steps, pretrain_skew)

    model = network_model.lstm_pretrain_model(bins_num, 
                                            time_steps, 
                                            embedding_dim, 
                                            hidden_size)


def train():

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = highfreq_helper.get_rnn_predict_dataset(data, time_steps, 20)

    model = network_model.lstm_classification_model(bins_num, 
                                                    time_steps, 
                                                    embedding_dim, 
                                                    hidden_size, 
                                                    predict_class)

    for e in range(epochs):
        print '------------Training model-----------, epoch', e + 1
        shuffled_rank = np.random.permutation(train_x.shape[0])
        train_x = train_x[shuffled_rank]
        train_y = train_y[shuffled_rank]
        model.fit(train_x, train_y, batch_size=batch_size, 
            class_weight={0:0., 1:0.4, 2:0.2, 3:0.4}, validation_data=(val_x, val_y))
        model_name = 'epoch_' + str(e + 1) + '_predict.model'
        model.save(os.path.join(models_path, model_name))
        print 'model saved to', os.path.join(models_path, model_name)

if __name__ == '__main__':
    train()
    test('epoch_' + str(epoch) + '_predict.model')