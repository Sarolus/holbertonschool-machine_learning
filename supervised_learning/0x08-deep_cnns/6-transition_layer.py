#!/usr/bin/env python3
"""
   Transition Layer Module
"""
import tensorflow.keras as keras


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer.

    Args:
        X : The output from the previous layer.
        nb_filters : Integer representing the number of
        filters.
        compression : COmpression factor for the transition layer.

    Returns:
        The output of the transition layer and the number of
        filters within the output.
    """
    initializer = keras.initializers.HeNormal()

    batch_norm = keras.layers.BatchNormalization()(X)
    activation = keras.layers.Activation("relu")(batch_norm)

    conv_layer = keras.layers.Conv2D(
        filters=int(nb_filters * compression),
        kernel_size=1,
        padding="same",
        kernel_initializer=initializer
    )(activation)

    avg_pool_layer = keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv_layer)

    return avg_pool_layer, int(nb_filters * compression)
