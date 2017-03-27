from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, TimeDistributed

def mlp_model(input_dim, hidden_size, num_class, activation_func='tanh'):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=input_dim, activation=activation_func))
    model.add(Dense(hidden_size, activation=activation_func))
    model.add(Dense(num_class, activation='softmax'))
    return model

def lstm_model(input_dim, seq_len, hidden_size, num_class):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(seq_len, input_dim), return_sequences=True))
    model.add(TimeDistributed(Dense(hidden_size, activation='tanh')))
    model.add(TimeDistributed(Dense(num_class, activation='softmax')))
    return model