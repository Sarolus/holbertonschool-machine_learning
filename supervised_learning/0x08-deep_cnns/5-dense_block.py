#!/usr/bin/env python3
"""
    Dense Blocks Module
"""
import tensorflow.keras as keras


def dense_block(X, nb_filters, growth_rate, layers):
    """
        Builds a dense block

        Args:
            X : Output of the previous layer.
            nb_filters : Integer representing the number of
            filters in X.
            growth_rate : The growth rate for the dense block.
            layers : The number of layers in the dense block.

        Returns:
            The concatenated output of each layer within the
            Dense block and the number of filters.
    """
    initializer = keras.initializers.HeNormal()

    for i in range(layers):
        Y = keras.layers.BatchNormalization()(X)
        Y = keras.layers.Activation("relu")(Y)
        Y = keras.layers.Conv2D(
            filters=4*growth_rate,
            kernel_size=1,
            padding="same",
            kernel_initializer=initializer
        )(Y)
        Y = keras.layers.BatchNormalization()(Y)
        Y = keras.layers.Activation("relu")(Y)
        Y = keras.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding="same",
            kernel_initializer=initializer
        )(Y)

        X = keras.layers.concatenate([X, Y])
        nb_filters += growth_rate

    return X, nb_filters
