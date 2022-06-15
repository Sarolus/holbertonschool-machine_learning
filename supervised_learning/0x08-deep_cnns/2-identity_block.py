#!/usr/bin/env python3
"""
    Identity Blocks Module
"""
import tensorflow.keras as keras


def identity_block(A_prev, filters):
    """
        Builds an identity block.

        Args:
            A_prev : Output of the previous layer.
            filters : Tuple or list containing the filters.

        Returns:
            The activated output of the identity block.
    """
    F11, F3, F12 = filters

    initializer = keras.initializers.HeNormal()

    conv_layer_1 = keras.layers.Conv2D(
        filters=F11,
        kernel_size=1,
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

    batch_norm = keras.layers.BatchNormalization()(conv_layer_3)
    add = keras.layers.Add()([batch_norm, A_prev])
    activation = keras.layers.Activation("relu")(add)

    return activation
