# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.constraints import max_norm, unit_norm
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, merge, Embedding, Masking, Activation, Flatten, Dropout, concatenate
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization
from keras.layers import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from config import learning_rate, clip_norm, dropout_rate

adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, clipnorm=clip_norm)

def mlp_model(num_class, seq_len, embedd_dim, hidden_size, out_num, activation_func='tanh'):
    main_input = Input(shape=(seq_len,), name='main_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(main_input)
    flattened = Flatten()(embedding)
    output_layer = Dense(out_num+1, activation='softmax', name='main_output')(main_input)

    model = Model(input=main_input, output=output_layer)
    model.compile(optimizer='rmsprop', 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def cnn_model(num_class, seq_len, embedd_dim, filters, kernel_size, out_num, activation_func='tanh'):
    
    main_input = Input(shape=(seq_len,), name='main_input')
    vol_input = Input(shape=(seq_len,), name='vol_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(main_input)
    embedding_vol = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(vol_input)
    conv_1 = Conv1D(filters, kernel_size, padding='causal', activation='linear', strides=1)(embedding)
    # conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('tanh')(conv_1)

    conv_vol = Conv1D(filters, kernel_size, padding='causal', activation='linear', strides=1)(embedding_vol)
    # conv_vol = BatchNormalization()(conv_vol)
    conv_vol = Activation('tanh')(conv_vol)
    # day_pooling = MaxPooling1D(16)(conv_2)
    # flattened = Flatten()(day_pooling)
    flattened = GlobalMaxPooling1D()(conv_1)
    flattened_vol = GlobalMaxPooling1D()(conv_vol)
    concat = concatenate([flattened, flattened_vol], axis=-1)
    # flattened = GRU(32, dropout=dropout_rate)(conv_2)
    flattened = Dense(filters // 2, activation='tanh')(concat)
    flattened = Dropout(dropout_rate)(flattened)
    output_layer = Dense(out_num + 1, activation='softmax', name='main_output')(flattened)
    model = Model(input=[main_input, vol_input], output=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def rnn_pretrain_model(num_class, seq_len, embedd_dim, hidden_size, activation_func='tanh'):
    main_input = Input(shape=(seq_len,), dtype='int32', name='main_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=True)(main_input)
    rnn_mid = LSTM(hidden_size, return_sequences=True)(embedding)
    rnn_out = LSTM(hidden_size, return_sequences=True)(rnn_mid)
    output_layer = TimeDistributed(Dense(num_class+1, activation='softmax', name='main_output'), name='output')(rnn_out)

    model = Model(input=main_input, output=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def rnn_factor_model(seq_len, feature_in, hidden_size):
    main_input = Input(shape=(seq_len, feature_in),  name='main_input')
    rnn_out = GRU(hidden_size, return_sequences=False)(main_input)
    output_layer = Dense(3, activation='softmax')(rnn_out)
    model = Model(input=main_input, output=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def rnn_classification_model(num_class, seq_len, embedd_dim, hidden_size, out_num, activation_func='tanh'):
    main_input = Input(shape=(seq_len,), name='main_input')
    vol_input = Input(shape=(seq_len,), name='vol_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(main_input)
    embedding_vol = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(vol_input)
    # rnn_sequence = GRU(hidden_size, dropout=dropout_rate, return_sequences=True)(embedding)
    rnn_out = GRU(hidden_size, dropout=dropout_rate, return_sequences=False)(embedding)
    rnn_vol = GRU(hidden_size, dropout=dropout_rate, return_sequences=False)(embedding_vol)
    concat = concatenate([rnn_out, rnn_vol], axis=-1)
    # hidden_after_rnn = Dense(hidden_size, activation=activation_func)(lstm_out)
    output_layer = Dense(out_num+1, activation='softmax')(concat)

    model = Model(input=[main_input, vol_input], output=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model