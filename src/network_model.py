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
    output_layer = Dense(out_num+1, activation='softmax', name='main_output')(flattened)

    model = Model(inputs=main_input, outputs=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def cnn_model(num_class, seq_len, embedd_dim, filters, kernel_size, out_num, activation_func='tanh'):
    
    main_input = Input(shape=(seq_len,), name='main_input')
    vol_input = Input(shape=(seq_len,), name='vol_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(main_input)
    embedding_vol = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(vol_input)
    
    # value
    conv_1 = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(embedding)
    conv_1 = Dropout(dropout_rate)(conv_1)
    conv_1 = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(conv_1)
    conv_1 = Dropout(dropout_rate)(conv_1)
    conv_1 = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(conv_1)
    conv_1 = Dropout(dropout_rate)(conv_1)
    # conv_1 = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(conv_1)
    # conv_1 = Dropout(dropout_rate)(conv_1)

    # volume
    conv_vol = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(embedding_vol)
    conv_vol = Dropout(dropout_rate)(conv_vol)
    conv_vol = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(conv_vol)
    conv_vol = Dropout(dropout_rate)(conv_vol)
    conv_vol = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(conv_vol)
    conv_vol = Dropout(dropout_rate)(conv_vol)
    # conv_vol = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(conv_vol)
    # conv_vol = Dropout(dropout_rate)(conv_vol)

    flattened = GlobalMaxPooling1D()(conv_1)
    flattened_vol = GlobalMaxPooling1D()(conv_vol)
    concat = concatenate([flattened, flattened_vol], axis=-1)
    flattened = Dense(filters // 2, activation='tanh')(concat)
    flattened = Dropout(dropout_rate)(concat)
    output_layer = Dense(out_num + 1, activation='softmax', name='main_output')(flattened)
    model = Model(inputs=[main_input, vol_input], outputs=output_layer)
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

    model = Model(inputs=main_input, outputs=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def rnn_factor_model(seq_len, feature_in, hidden_size):
    main_input = Input(shape=(seq_len, feature_in),  name='main_input')
    bn = BatchNormalization()(main_input)
    rnn_out = LSTM(hidden_size, dropout=dropout_rate, return_sequences=False)(bn)
    # rnn_out = LSTM(hidden_size, dropout=dropout_rate, return_sequences=True)(rnn_out)
    # rnn_out = LSTM(hidden_size, dropout=dropout_rate, return_sequences=True)(rnn_out)
    # rnn_out = LSTM(hidden_size, dropout=dropout_rate, return_sequences=False)(rnn_out)
    output_layer = Dense(3, activation='softmax')(rnn_out)
    model = Model(inputs=main_input, outputs=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def cnn_factor_model(seq_len, feature_in, filters, kernel_size=12):
    main_input = Input(shape=(seq_len, feature_in), name='main_input')
    bn = BatchNormalization()(main_input)
    conv = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(main_input)
    # conv = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(conv)
    # conv = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(conv)
    # conv = Conv1D(filters, kernel_size, padding='causal', activation='tanh', strides=1)(conv)
    flattened = GlobalMaxPooling1D()(conv)
    flattened = Dropout(dropout_rate)(flattened)
    output_layer = Dense(3, activation='softmax')(flattened)
    model = Model(inputs=main_input, outputs=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def mlp_factor_model(seq_len, feature_in):
    main_input = Input(shape=(seq_len, feature_in), name='main_input')
    bn = BatchNormalization()(main_input)
    flattened = Flatten()(bn)
    output_layer = Dense(3, activation='softmax')(flattened)
    model = Model(inputs=main_input, outputs=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

def rnn_classification_model(num_class, seq_len, embedd_dim, hidden_size, out_num, activation_func='tanh'):
    main_input = Input(shape=(seq_len,), name='main_input')
    vol_input = Input(shape=(seq_len,), name='vol_input')
    embedding = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(main_input)
    embedding_vol = Embedding(output_dim=embedd_dim, input_dim=num_class+1, input_length=seq_len, mask_zero=False)(vol_input)

    # values
    rnn_out = LSTM(hidden_size, dropout=dropout_rate, return_sequences=False)(embedding)
    
    # volume
    rnn_vol = LSTM(hidden_size, dropout=dropout_rate, return_sequences=False)(embedding_vol)
    
    concat = concatenate([rnn_out, rnn_vol], axis=-1)
    output_layer = Dense(out_num+1, activation='softmax')(concat)
    model = Model(inputs=[main_input, vol_input], outputs=output_layer)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model