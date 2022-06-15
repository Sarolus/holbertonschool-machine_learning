#!/usr/bin/env python3
"""
    Projection Blocks Module
"""
import tensorflow.keras as keras


def projection_block(A_prev, filters, s=2):
    """
        Builds an projection block.

        Args:
            A_prev : Output of the previous layer.
            filters : Tuple or list containing the filters.
            s : stride of the first convolution in both the main
            path and the shortcut connection.

        Returns:
            The activated output of the projection block.
    """
    F11, F3, F12 = filters

    initializer = keras.initializers.HeNormal()

    conv_layer_1 = keras.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=s,
        padding="same",
        kernel_initializer=initializer
    )(A_prev)

    batch_norm = keras.layers.BatchNormalization()(conv_layer_1)
    activation = keras.layers.Activation("relu")(batch_norm)

    conv_layer_2 = keras.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding="same",
        kernel_initializer=initializer
    )(activation)

    batch_norm = keras.layers.BatchNormalization()(conv_layer_2)
    activation = keras.layers.Activation("relu")(batch_norm)

    conv_layer_3 = keras.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding="same",
        kernel_initializer=initializer
    )(activation)

    main_batch_norm = keras.layers.BatchNormalization()(conv_layer_3)

    conv_layer_4 = keras.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=s,
        padding="same",
        kernel_initializer=initializer
    )(A_prev)

    short_batch_norm = keras.layers.BatchNormalization()(conv_layer_4)

    add = keras.layers.Add()([main_batch_norm, short_batch_norm])
    activation = keras.layers.Activation("relu")(add)

    return activation
