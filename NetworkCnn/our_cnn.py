import keras
from keras import layers
from keras import models
from keras.optimizers import Adam, SGD
from Dataset.Loaddata import LoadData
from sklearn.metrics import classification_report
from NetworkCnn.Network import create_res_net
from NetworkCnn.vgg16 import bulid_vgg16
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten,Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

class Our_cnn:
    @staticmethod
    def build_model(inputshape=28):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=1024, kernel_size=1, padding='same',
                                input_shape=(inputshape, 1)))
        model.add(layers.Reshape(input_shape=(28,1024),target_shape=(28,32,32)))
        model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

        # model.add(layers.Conv1D(filters=filters[1], kernel_size=3, padding='same', activation='relu',kernel_regularizer=l2(regularizer)))
        #model.add(layers.MaxPooling2D(pool_size=2))
        #model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
        #model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
        #model.add(MaxPool2D(pool_size=2))
        # model.add(Dropout(0.1))
        #model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
        #model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
        #model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
        #model.add(MaxPool2D(pool_size=2))
        # model.add(Dropout(0.1))
        #model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        #model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        #model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        #model.add(MaxPool2D(pool_size=2))
        #model.add(Dropout(0.25))
        #model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        #model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        #model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        #model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())
        #model.add(Dropout(0.1))
        model.add(Dense(units=1024, activation="relu", kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.25))
        model.add(Dense(units=1024, activation="relu", kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.25))
        model.add(Dense(units=2, activation="softmax"))
        return model