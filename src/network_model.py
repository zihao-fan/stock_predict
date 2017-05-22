# -*- coding: utf-8 -*-

from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.constraints import max_norm, unit_norm
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed, merge, Embedding, Masking, Activation, Flatten, Conv1D
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from config import learning_rate, clip_norm, dropout_rate

def mlp_model(num_class, seq_len, embedd_dim, hidden_size, out_num, activation_func='tanh'):
    main_input = Input(shape=(seq_len,), dtype='int32', name='main_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(main_input)
    flattened = Flatten()(embedding)
    output_layer = Dense(out_num+1, activation='softmax', name='main_output')(flattened)

    model = Model(input=main_input, output=output_layer)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def cnn_model(num_class, seq_len, embedd_dim, hidden_size, out_num, activation_func='tanh'):
    main_input = Input(shape=(seq_len,), dtype='int32', name='main_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(main_input)
    conv_1 = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(embedding)
    conv_2 = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)

def lstm_pretrain_model(num_class, seq_len, embedd_dim, hidden_size, activation_func='tanh'):
    main_input = Input(shape=(seq_len,), dtype='int32', name='main_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=True)(main_input)
    lstm_mid = LSTM(hidden_size, return_sequences=True)(embedding)
    lstm_out = LSTM(hidden_size, return_sequences=True)(lstm_mid)
    output_layer = TimeDistributed(Dense(num_class+1, activation='softmax', name='main_output'), name='output')(lstm_out)

    model = Model(input=main_input, output=output_layer)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def lstm_classification_model(num_class, seq_len, embedd_dim, hidden_size, out_num, activation_func='tanh'):
    main_input = Input(shape=(seq_len,), dtype='int32', name='main_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=True)(main_input)
    # lstm_sequence = LSTM(hidden_size, dropout=dropout_rate, return_sequences=True)(embedding)
    lstm_out = LSTM(hidden_size, dropout=dropout_rate, return_sequences=False)(embedding)
    # hidden_after_rnn = Dense(hidden_size, activation=activation_func)(lstm_out)
    output_layer = Dense(out_num+1, activation='softmax')(lstm_out)

    model = Model(input=main_input, output=output_layer)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    return model