# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import AveragePooling2D, AveragePooling1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
#from keras import backend as K

class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, stride, padding="same"):
        # define a CONV => BN => RELU pattern
        x = Conv1D(K, kX, strides=stride, padding=padding)(x)
        #x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        # return the block
        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3):
        # define two CONV modules, then concatenate across the
        # channel dimension
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3)
        x = concatenate([conv_1x1, conv_3x3], axis=0)
        # return the block
        return x

    @staticmethod
    def downsample_module(x, K):
        # define the CONV module and POOL, then concatenate
        # across the channel dimensions
        conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 2)
        pool = MaxPooling1D(3, strides= 2)(x)
        x = concatenate([conv_3x3, pool], axis=0)
        # return the block
        return x

    @staticmethod
    def build(width=28, height=1, classes=2):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (width,height)
        # if we are using "channels first", update the input shape
        # and channels dimension
        # define the model input and first CONV module
        inputs = Input(shape=inputShape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 1)
        # two Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 32, 32)
        x = MiniGoogLeNet.inception_module(x, 32, 48)
        x = MiniGoogLeNet.downsample_module(x, 80)
        # four Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 112, 48)
        x = MiniGoogLeNet.inception_module(x, 96, 64)
        x = MiniGoogLeNet.inception_module(x, 80, 80)
        x = MiniGoogLeNet.inception_module(x, 48, 96)
        x = MiniGoogLeNet.downsample_module(x, 96)
        # two Inception modules followed by global POOL and dropout
        x = MiniGoogLeNet.inception_module(x, 176, 160)
        x = MiniGoogLeNet.inception_module(x, 176, 160)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)
        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)
        # create the model
        model = Model(inputs, x, name="googlenet")
        # return the constructed network architecture
        return model