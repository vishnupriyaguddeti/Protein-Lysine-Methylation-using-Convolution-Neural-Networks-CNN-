import keras 

class InceptionModel(keras.layers.Layer):
    def __init__(self, num_filters=32, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        #self.activation = keras.activations.get(activation)

    def _default_Conv1D(self, filters, kernel_size):
        return keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                   strides=1, activation='relu',use_bias=False)

    def call(self, inputs):
        #step1
        z_bottleneck = self._default_Conv1D(filters=self.num_filters, kernel_size=1)(inputs)
        z_maxpool = keras.layers.MaxPooling1D(pool_size=3, strides=1,
                                              padding='same')(inputs)
        # step 2
        z1 = self._default_Conv1D(filters=self.num_filters,kernel_size=10)(z_bottleneck)
        z2 = self._default_Conv1D(filters=self.num_filters, kernel_size=20)(z_bottleneck)
        z3 = self._default_Conv1D(filters=self.num_filters, kernel_size=40)(z_bottleneck)
        z4 = self._default_Conv1D(filters=self.num_filters, kernel_size=1)(z_maxpool)

        # step 3
        z = keras.layers.Concatenate(axis=2)([z1, z2, z3, z4])
        z = keras.layers.BatchNormalization()(z)
        return self.activation(z)

def shortcut_layer(inputs, z_inception):
    # create shortcut to inception
    z_shortcut = keras.layers.Conv1D(filters=int(inputs.shape[-1]), kernel_size=1, padding='same',
                                     use_bias=False)(inputs)

    z_shortcut = keras.layers.BatchNormalization()(z_shortcut)
    z = keras.layers.Add()([z_shortcut, z_inception])
    return keras.layers.activations('relu')(z)

def  build_Model(input_shape=(1,28), num_classes=2, num_models=6):
    input_layer = keras.layers.Input(input_shape)
    z = input_layer
    z_residual = input_layer

    for i in range(num_models):

        z = InceptionModel()(z)
        if i % 3 == 2:
            z = shortcut_layer(z_residual,z)
            z_residual = z
    gap_layer = keras.layers.GlobalAveragePooling1D()(z)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model