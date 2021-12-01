from keras import Model, Input, layers
from keras.layers import Conv2D, Conv1D, Reshape, MaxPooling1D, Dropout, concatenate, Flatten, Dense, MaxPooling2D
def build_ensable(input_shape=28):
    inputs = Input(shape=(input_shape, 1), name="original_img")


    x = Conv1D(filters=1024, kernel_size=1, padding='same')(inputs)
    x = Reshape(input_shape=(28,1024),target_shape=(28,32,32))(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)

    y = Conv1D(filters=32,kernel_size=3,padding='same',activation='relu')(inputs)
    y = Conv1D(filters=32,kernel_size=3,padding='same',activation='relu')(y)
    y = Conv1D(filters=32,kernel_size=3,padding='same',activation='relu')(y)
    y = MaxPooling1D(pool_size=2)(y)

    y = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(y)
    y = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(y)
    y = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(y)
    y = MaxPooling1D(pool_size=2)(y)

    y = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(y)
    y = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(y)
    y = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(y)
    #model.add(layers.MaxPooling1D(pool_size=2))

    y = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(y)
    y = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(y)
    y = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(y)
    y = Dropout(0.25)(y)

    #model.add(layers.MaxPooling1D(pool_size=2))
    y = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(y)
    y = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(y)
    y = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(y)

    y = Flatten()(y)
    y = Dense(4096, activation='relu')(y)

    z = concatenate([x,y])

    #z = Flatten()(z)
    # model.add(layers.Dense(512,activation='relu'))

    z = Dense(1024, activation='relu')(z)
    z = Dense(512, activation='relu')(z)
    z = Dense(2, activation='softmax')(z)

    model = Model(inputs=[inputs], outputs=[z])


    return model









