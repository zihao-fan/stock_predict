# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import os
import network_model
import highfreq_helper
from highfreq_helper import root_path
from keras.models import load_model
from config import batch_size, epochs, time_steps, embedding_dim, hidden_size, kernel_size, filters
from config import INPUT_SKEW, INPUT_BIN
from config import OUTPUT_SKEW, OUTPUT_BIN
from scipy.stats import normaltest

models_path = os.path.join(root_path, 'models')
results_path = os.path.join(root_path, 'results')

# datafile_path = os.path.join(root_path, 'data', '000905_20100101_20170525.data')
# data = highfreq_helper.read_pickle(datafile_path)

datafile_path = os.path.join(root_path, 'data', '000905_20100101_20170524.data')
data = highfreq_helper.read_pickle(datafile_path)

def evaluate(prediction, label):
    total_number = 0
    hit_number = 0
    
    rise_fall_total = 0
    rise_fall_hit = 0

    market_hit = 0

    for i in range(len(prediction)):
        # overall result
        total_number += 1
        if prediction[i] == label[i]:
            hit_number += 1

        # if (prediction[i] != 2) == (label[i] != 2):
        #     market_hit += 1
        
        # # rise fall result
        # if label[i] != 2 and prediction[i] != 2: # rise or fall
        #     rise_fall_total += 1
        #     if prediction[i] == label[i]:
        #         rise_fall_hit += 1
        
    # return float(hit_number) / total_number, float(rise_fall_hit) / rise_fall_total, float(market_hit) / total_number
    return float(hit_number) / total_number

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

def test(model_name, day_num=10):
    (train_x, train_vol, train_y), (val_x, val_vol, val_y), (test_x, test_vol, test_y) = highfreq_helper.get_rnn_predict_dataset(data, 
                                                                            day_num * time_steps, OUTPUT_SKEW)

    print('test_x shape', test_x.shape, 'test_y shape', test_y.shape)
    model = load_model(os.path.join(models_path, model_name))
    
    print('Testing the model')
    prediction = model.predict([test_x, test_vol])
    # prediction = model.predict(test_x)
    max_confidence = np.max(prediction, axis=1)

    train_labels = np.argmax(train_y, axis=-1)
    labels = np.argmax(test_y, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    train_stats = get_stats_dict(train_labels)
    prediction_stats = get_stats_dict(prediction)
    label_stats = get_stats_dict(labels)
    print('train label stats', train_stats)
    print('prediciton stats', prediction_stats)
    print('label stats', label_stats)

    numbers = len(max_confidence)
    sorted_index = np.argsort(-max_confidence)
    
    print("All predictions")
    acc = evaluate(prediction, labels)
    print('acc', acc)

    prediction = prediction[sorted_index]
    labels = labels[sorted_index]
    print("Top 10%")
    acc = evaluate(prediction[:int(0.1*numbers)], labels[:int(0.1*numbers)])
    print('acc', acc)

def pretrain():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = highfreq_helper.get_rnn_pretrain_dataset(data, 
                                                                        time_steps, INPUT_SKEW)

    model = network_model.rnn_pretrain_model(INPUT_BIN, 
                                            time_steps, 
                                            embedding_dim, 
                                            hidden_size)


def train_rnn(day_num=1):

    (train_x, train_vol, train_y), (val_x, val_vol, val_y), (test_x, test_vol, test_y) = highfreq_helper.get_rnn_predict_dataset(data, 
                                                                            day_num * time_steps, OUTPUT_SKEW)

    model = network_model.rnn_classification_model(INPUT_BIN, 
                                                    day_num * time_steps, 
                                                    embedding_dim, 
                                                    hidden_size, 
                                                    OUTPUT_BIN)

    for e in range(epochs):
        print('------------Training model-----------, epoch', e + 1)
        shuffled_rank = np.random.permutation(train_x.shape[0])
        train_x = train_x[shuffled_rank]
        train_vol = train_vol[shuffled_rank]
        train_y = train_y[shuffled_rank]
        model.fit([train_x, train_vol], train_y, batch_size=batch_size, 
            # class_weight={0:0., 1:0.38, 2:0.24, 3:0.38}, 
            validation_data=([val_x, val_vol], val_y))
        model_name = 'epoch_' + str(e + 1) + '_predict.model'
        model.save(os.path.join(models_path, model_name))
        print('model saved to', os.path.join(models_path, model_name))

def train_mlp(day_num=10):

    (train_x, train_vol, train_y), (val_x, val_vol, val_y), (test_x, test_vol, test_y) = highfreq_helper.get_rnn_predict_dataset(data, 
                                                                            day_num * time_steps, OUTPUT_SKEW)
    print('train_x shape', train_x.shape, 'train_y shape', train_y.shape)
    model = network_model.mlp_model(INPUT_BIN,
                                    day_num * time_steps, 
                                    embedding_dim,
                                    hidden_size,
                                    OUTPUT_BIN)

    for e in range(epochs):
        print('------------Training model-----------, epoch', e + 1)
        shuffled_rank = np.random.permutation(train_x.shape[0])
        train_x = train_x[shuffled_rank]
        train_y = train_y[shuffled_rank]
        model.fit(train_x, train_y, batch_size=batch_size, 
            # class_weight={0:0., 1:0.38, 2:0.24, 3:0.38}, 
            validation_data=(val_x, val_y))
        model_name = 'epoch_' + str(e + 1) + '_predict.model'
        model.save(os.path.join(models_path, model_name))
        print('model saved to', os.path.join(models_path, model_name))

def train_cnn(day_num=10):

    (train_x, train_vol, train_y), (val_x, val_vol, val_y), (test_x, test_vol, test_y) = highfreq_helper.get_rnn_predict_dataset(data, 
                                                                            day_num * time_steps, OUTPUT_SKEW)

    print('train_x shape', train_x.shape, 'train_vol shape', train_vol.shape, 'train_y shape', train_y.shape)
    model = network_model.cnn_model(INPUT_BIN,
                                    day_num * time_steps, 
                                    embedding_dim,
                                    filters,
                                    kernel_size,
                                    OUTPUT_BIN)

    for e in range(epochs):
        print('------------Training model-----------, epoch', e + 1)
        shuffled_rank = np.random.permutation(train_x.shape[0])
        train_x = train_x[shuffled_rank]
        train_vol = train_vol[shuffled_rank]
        train_y = train_y[shuffled_rank]
        model.fit([train_x, train_vol], train_y, batch_size=batch_size, 
            # class_weight={0:0., 1:0.38, 2:0.24, 3:0.38}, 
            validation_data=([val_x, val_vol], val_y))
        model_name = 'epoch_' + str(e + 1) + '_predict.model'
        model.save(os.path.join(models_path, model_name))
        print('model saved to', os.path.join(models_path, model_name))

def train_real_value(mins=120, label='ascent_hour'):

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = highfreq_helper.generate_rnn_data(data, mins, label)
    print('train_x shape', train_x.shape, 'train_y shape', train_y.shape)
    model = network_model.rnn_realvalue_model(mins, train_x.shape[2], hidden_size)

    for e in range(epochs):
        print('------------Training model-----------, epoch', e + 1)
        shuffled_rank = np.random.permutation(train_x.shape[0])
        train_x = train_x[shuffled_rank]
        train_y = train_y[shuffled_rank]
        model.fit(train_x, train_y, batch_size=batch_size, 
            # class_weight={0:0., 1:0.38, 2:0.24, 3:0.38}, 
            validation_data=(val_x, val_y))
        model_name = 'epoch_' + str(e + 1) + '_predict.model'
        model.save(os.path.join(models_path, model_name))
        print('model saved to', os.path.join(models_path, model_name))

if __name__ == '__main__':
    days = 20
    # array = data['rate_1'].as_matrix()
    # res1, res2 = normaltest(np.asarray([1.0]*1000))
    # print(res1, res2)
    # train_rnn(days)
    # train_mlp(days)
    # train_cnn(days)
    # train_real_value()
    # test('epoch_' + str(epochs) + '_predict.model', days)