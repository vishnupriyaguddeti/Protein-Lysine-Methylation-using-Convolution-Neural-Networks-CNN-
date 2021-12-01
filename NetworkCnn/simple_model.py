import keras
from keras import layers
from keras import models
from keras.regularizers import l2
from keras import Input

def build_cnn(inputshape=28,filters=[512,256,128,64,64],regularizer=None):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=filters[0],kernel_size=3,padding='same',activation='relu',input_shape=(inputshape,1)))
    model.add(layers.Conv1D(filters=filters[0],kernel_size=3,padding='same',activation='relu'))
    #model.add(layers.Conv1D(filters=filters[0],kernel_size=3,padding='same',activation='relu'))
    #model.add(layers.Conv1D(filters=filters[0],kernel_size=3,padding='same',activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=filters[1], kernel_size=3, padding='same', activation='relu'))
    #model.add(layers.Conv1D(filters=filters[1], kernel_size=3, padding='same', activation='relu'))
   # model.add(layers.Conv1D(filters=filters[1], kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=filters[2], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(0.01)))
    #model.add(layers.Conv1D(filters=filters[2], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(0.01)))
    #model.add(layers.Conv1D(filters=filters[2], kernel_size=7, padding='same', activation='relu'))
    #model.add(layers.Conv1D(filters=filters[2], kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=filters[3], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(0.01)))
    #model.add(layers.Conv1D(filters=filters[3], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(0.01)))
    #model.add(layers.Conv1D(filters=filters[3], kernel_size=3, padding='same', activation='relu'))
    #model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=filters[4], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(0.01)))
    #model.add(layers.Conv1D(filters=filters[4], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(0.01)))
    #model.add(layers.MaxPooling1D(pool_size=2))
    #model.add(layers.Conv1D(filters=filters[4], kernel_size=3, padding='same', activation='relu'))
    #model.add(layers.Conv1D(filters=filters[4], kernel_size=3, padding='same', activation='relu'))
    #model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())
    #model.add(layers.Dense(1024,activation='relu'))
    #model.add(layers.Dense(512,activation='relu'))
    #model.add(layers.Dense(512,activation='relu'))
    #model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()
    return model

def build_cnn_api(inputshape=28,filters=[512,256,128,64,64],regularizer=None):
    inputs = Input(shape=(inputshape,1))
    x = layers.Conv1D(filters=filters[0],kernel_size=3,padding='same',activation='relu')(inputs)
    x =  layers.Conv1D(filters=filters[0],kernel_size=3,padding='same',activation='relu')(x)
    x =  layers.MaxPooling1D(pool_size=2)(x)

    x1 = layers.Conv1D(filters=filters[1], kernel_size=3, padding='same', activation='relu')(x)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)

    x2 = layers.Conv1D(filters=filters[2], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(0.01))(x1)
    x2 = layers.MaxPooling1D(pool_size=2)(x2)

    x3 = layers.Conv1D(filters=filters[3], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(0.01))(x2)
    x3 = layers.MaxPooling1D(pool_size=2)(x3)

    x4 = layers.Conv1D(filters=filters[4], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(0.01))(x3)

    x5 = layers.Flatten()(x4)
    x5 = layers.Dense(64, activation='relu')(x5)
    x5 = layers.Dense(2, activation='softmax')(x5)
    model = models.Model(inputs=inputs, outputs=x5)
    return model






