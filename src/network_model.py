# -*- coding: utf-8 -*-

from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import Input, Dense, LSTM, TimeDistributed, merge, Embedding, Masking, Activation

def mlp_model(input_dim, seq_len, hidden_size, num_class, activation_func='tanh'):

    input_layer = Input(shape=(input_dim*seq_len,), name='MLP Input Layer')
    hidden_layer_1 = Dense(hidden_size, activation=activation_func)(input_layer)
    hidden_layer_2 = Dense(hidden_size, activation=activation_func)(hidden_layer_1)
    output_layer = Dense(num_class, activation='softmax', name='softmax_out')(hidden_layer_2)

    model = Model(input=input_layer, output=output_layer)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    model.summary()
    return model

def lstm_model(input_dim, seq_len, hidden_size, num_class, activation_func='tanh'):

    input_layer = Input(shape=(seq_len, input_dim), name='RNN Input Layer')
    lstm_sequence = LSTM(hidden_size, return_sequences=True)(input_layer)
    hidden_after_rnn = TimeDistributed(Dense(hidden_size, activation=activation_func), name='hidden_after_rnn')(lstm_sequence)
    output_layer = TimeDistributed(Dense(num_class, activation='softmax'), name='softmax_output')(hidden_after_rnn)

    model = Model(input=input_layer, output=output_layer)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    model.summary()

    return model