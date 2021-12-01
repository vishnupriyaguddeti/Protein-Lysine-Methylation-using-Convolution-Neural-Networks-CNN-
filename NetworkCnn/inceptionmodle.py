from keras.models import Model
from keras.layers import Conv1D, MaxPool1D, \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling1D, AveragePooling1D,\
    Flatten
from  keras.regularizers import l2

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    conv_1x1 = Conv1D(filters_1x1, 1, padding='same', activation='relu')(x)

    conv_3x3 = Conv1D(filters_3x3_reduce, 1, padding='same', activation='relu')(x)
    conv_3x3 = Conv1D(filters_3x3, 3, padding='same', activation='relu')(conv_3x3)

    conv_5x5 = Conv1D(filters_5x5_reduce, 1, padding='same', activation='relu')(x)
    conv_5x5 = Conv1D(filters_5x5, 5, padding='same', activation='relu')(conv_5x5)

    pool_proj = MaxPool1D(3, strides=1, padding='same')(x)
    pool_proj = Conv1D(filters_pool_proj, 1, padding='same', activation='relu')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=2, name=name)

    return output
def build_inception(input=28):
    input_layer = Input(shape=(input, 1))

    x = Conv1D(64, 3, padding='same', strides=1, activation='relu', name='conv_1_7x7/2')(input_layer)
    #x = MaxPool1D(3, padding='same', strides=1, name='max_pool_1_3x3/2')(x)
    #x = Conv1D(64, 1, padding='same', strides=1, activation='relu', name='conv_2a_3x3/1')(x)
    #x = Conv1D(192, 3, padding='same', strides=1, activation='relu', name='conv_2b_3x3/1')(x)
    #x = MaxPool1D(3, padding='same', strides=1, name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=32,
                         filters_3x3_reduce=96,
                         filters_3x3=32,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')

    x = inception_module(x,
                         filters_1x1=32,
                         filters_3x3_reduce=96,
                         filters_3x3=32,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3b')

    x = MaxPool1D(2, padding='same', strides=2, name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=192,
                         filters_3x3=64,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4a')


    #x1 = AveragePooling1D(5, strides=3)(x)
    #x1 = Conv1D(128, 1, padding='same', activation='relu')(x1)
    #x1 = Flatten()(x1)
    #x1 = Dense(1024, activation='relu',activity_regularizer=l2(0.001))(x1)
    #x1 = Dropout(0.7)(x1)
    #x1 = Dense(64, activation='relu', name='auxilliary_output_1')(x1)

    #x = inception_module(x,
     #                    filters_1x1=16,
     #                    filters_3x3_reduce=48,
     #                    filters_3x3=16,
     #                    filters_5x5_reduce=8,
     #                    filters_5x5=16,
     #                    filters_pool_proj=16,
     #                    name='inception_4b')

    #x = inception_module(x,
     #                    filters_1x1=32,
     #                    filters_3x3_reduce=96,
     #                    filters_3x3=32,
     #                    filters_5x5_reduce=16,
     #                    filters_5x5=32,
      #                   filters_pool_proj=32,
     #                    name='inception_4c')

    #x = inception_module(x,
     #                    filters_1x1=32,
     #                    filters_3x3_reduce=96,
     #                    filters_3x3=32,
     #                    filters_5x5_reduce=16,
      #                   filters_5x5=32,
      #                   filters_pool_proj=32,
      #                   name='inception_4d')


    #x2 = AveragePooling1D(5, strides=3)(x)
    #x2 = Conv1D(128, 1, padding='same', activation='relu')(x2)
    #x2 = Flatten()(x2)
    #x2 = Dense(1024, activation='relu',activity_regularizer=l2(0.001))(x2)
    #x2 = Dropout(0.7)(x2)
    #x2 = Dense(64, activation='relu', name='auxilliary_output_2')(x2)

    #x = inception_module(x,
     #                    filters_1x1=64,
     #                    filters_3x3_reduce=192,
      #                   filters_3x3=64,
     #                    filters_5x5_reduce=32,
      #                   filters_5x5=64,
      #                   filters_pool_proj=64,
      #                   name='inception_4e')

    #x = MaxPool1D(3, padding='same', strides=2, name='max_pool_4_3x3/2')(x)

    #x = inception_module(x,
      #                   filters_1x1=64,
       #                  filters_3x3_reduce=192,
       #                  filters_3x3=64,
       #                  filters_5x5_reduce=32,
       #                  filters_5x5=64,
        #                 filters_pool_proj=64,
        #                 name='inception_5a')

    #x = inception_module(x,
      #                   filters_1x1=64,
     #                    filters_3x3_reduce=192,
      #                   filters_3x3=64,
      #                   filters_5x5_reduce=32,
       #                  filters_5x5=64,
        #                 filters_pool_proj=64,
        #                 name='inception_5b')

    x = GlobalAveragePooling1D(name='avg_pool_5_3x3/1')(x)

    #x = Dropout(0.50)(x)

    x = Dense(512, activation='relu', name='output1')(x)
    #z = concatenate([x,x1])
    z = Dense(2, activation='softmax', name='output')(x)

    model = Model(input_layer, z, name='inception_v1')
    return model