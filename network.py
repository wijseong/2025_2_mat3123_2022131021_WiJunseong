# network.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
import config

def create_model(n_vocab, input_shape):
    """LSTM 기반의 음악 생성 모델을 정의합니다."""
    model = Sequential()
    
    model.add(LSTM(config.LSTM_UNITS, 
                   input_shape=input_shape, 
                   return_sequences=True))
    model.add(Dropout(config.DROPOUT_RATE))
    
    model.add(LSTM(config.LSTM_UNITS))
    model.add(Dropout(config.DROPOUT_RATE))
    
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    return model