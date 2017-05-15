# -*- coding: utf-8 -*-

from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.constraints import max_norm, unit_norm
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed, merge, Embedding, Masking, Activation
from keras.layers.normalization import BatchNormalization
from config import learning_rate, clip_norm

def mlp_model(input_dim, seq_len, hidden_size, num_class, activation_func='tanh'):
    input_layer = Input(shape=(input_dim*seq_len,))
    bn = BatchNormalization()(input_layer)
    hidden_layer_1 = Dense(hidden_size, activation=activation_func)(bn)
    hidden_layer_2 = Dense(hidden_size, activation=activation_func)(hidden_layer_1)
    output_layer = Dense(num_class, activation='softmax', name='softmax_out')(hidden_layer_2)

    model = Model(input=input_layer, output=output_layer)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, clipnorm=clip_norm)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model

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

def lstm_classification_model(input_dim, seq_len, hidden_size, num_class, activation_func='tanh'):
    input_layer = Input(shape=(seq_len, input_dim))
    lstm_sequence_1 = LSTM(hidden_size, return_sequences=True)(input_layer)
    lstm_out = LSTM(hidden_size, return_sequences=False)(lstm_sequence_1)
    hidden_after_rnn = Dense(hidden_size, activation=activation_func)(lstm_out)
    output_layer = Dense(num_class+1, activation='softmax')(hidden_after_rnn)

    model = Model(input=input_layer, output=output_layer)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    return model