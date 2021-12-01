import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D , Flatten,Dropout
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
def bulid_vgg16(inputshape=28):
    model = Sequential()
    model.add(Conv1D(input_shape=(inputshape, 1), filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))
    #model.add(Dropout(0.1))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))
    #model.add(Dropout(0.1))
    model.add(Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))
    #model.add(Dropout(0.1))
    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(units=4096, activation="relu",kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.25))
    model.add(Dense(units=4096, activation="relu",kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.25))
    model.add(Dense(units=2, activation="softmax"))
    return model