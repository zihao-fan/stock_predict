# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import os
import pickle
import network_model
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from factor_helper import root_path
from config import batch_size, epochs, time_steps, embedding_dim, hidden_size, kernel_size, filters

dataset_path = '/research/zihao/factor_dataset.cpickle'
models_path = os.path.join(root_path, 'models')
results_path = os.path.join(root_path, 'results')

(train_x, train_y, val_x, val_y, test_x, test_y) = pickle.load(open(dataset_path, 'rb'))
train_y = to_categorical(train_y)
val_y = to_categorical(val_y)
test_y = to_categorical(test_y)
print(np.isnan(train_x).any(), np.isnan(train_y).any())
print(train_x.shape, train_y.shape)

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

def test(model_name):
    print('test_x shape', test_x.shape, 'test_y shape', test_y.shape)
    model = load_model(os.path.join(models_path, model_name))
    
    print('Testing the model')
    prediction = model.predict(test_x)
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

def train_cnn():
    global train_x, train_y, val_x, val_y, test_x, test_y
    model = network_model.cnn_factor_model(24, 79, 100, 24)
    # model = network_model.mlp_factor_model(24, 79)
    for e in range(epochs):
        print('------------Training model-----------, epoch', e + 1)
        shuffled_rank = np.random.permutation(train_x.shape[0])
        train_x = train_x[shuffled_rank]
        train_y = train_y[shuffled_rank]
        model.fit(train_x, train_y, batch_size=batch_size,
            validation_data=(val_x, val_y))
        model_name = 'epoch_' + str(e + 1) + '_predict.model'
        model.save(os.path.join(models_path, model_name))
        print('model saved to', os.path.join(models_path, model_name))

def train_rnn():
    global train_x, train_y, val_x, val_y, test_x, test_y
    model = network_model.rnn_factor_model(24, 79, 100)
    for e in range(epochs):
        print('------------Training model-----------, epoch', e + 1)
        shuffled_rank = np.random.permutation(train_x.shape[0])
        train_x = train_x[shuffled_rank]
        train_y = train_y[shuffled_rank]
        model.fit(train_x, train_y, batch_size=batch_size,
            validation_data=(val_x, val_y))
        model_name = 'epoch_' + str(e + 1) + '_predict.model'
        model.save(os.path.join(models_path, model_name))
        print('model saved to', os.path.join(models_path, model_name))

if __name__ == '__main__':
    train_cnn()
    # train_rnn()
    test('epoch_' + str(epochs) + '_predict.model')