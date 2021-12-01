from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv1D, ReLU, BatchNormalization, \
    Add, AveragePooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from keras.regularizers import l2


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv1D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same",kernel_regularizer=l2(0.01))(x)
    y = relu_bn(y)
    y = Conv1D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same",kernel_regularizer=l2(0.01))(y)

    if downsample:
        x = Conv1D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same",kernel_regularizer=l2(0.01))(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def create_res_net(inputshape=28):
    inputs = Input(shape=(inputshape,1))
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same",kernel_regularizer=l2(0.01))(t)
    t = relu_bn(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling1D(4)(t)
    t = Dropout(0.9)(t)
    t = Flatten()(t)
    outputs = Dense(2, activation='softmax')(t)

    model = Model(inputs, outputs)

   # model.compile(
    #    optimizer='adam',
    #    loss='sparse_categorical_crossentropy',
    #    metrics=['accuracy']
    #)

    return model