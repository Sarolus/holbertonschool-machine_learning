#!/usr/bin/env python3
"""
   DenseNet-121 Architecture Module
"""
import tensorflow.keras as keras
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
        Builds the DenseNet-121 Architecture.

        Args:
            growth_rate (int, optional): The growth rate used.
            Defaults to 32.
            compression (float, optional): The compression factor.
            Defaults to 1.0.

        Returns:
            The keras model.
    """
    initializer = keras.initializers.HeNormal()

    X = keras.Input(shape=(224, 224, 3))

    Y = keras.layers.BatchNormalization()(X)
    Y = keras.layers.Activation("relu")(Y)

    Y = keras.layers.Conv2D(
        filters=2*growth_rate,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
    )(Y)

    Y = keras.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(Y)

    Y, nb_filters = dense_block(
        X=Y,
        nb_filters=2*growth_rate,
        growth_rate=growth_rate,
        layers=6,
    )

    Y, nb_filters = transition_layer(
        X=Y,
        nb_filters=nb_filters,
        compression=compression,
    )

    Y, nb_filters = dense_block(
        X=Y,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=12
    )

    Y, nb_filters = transition_layer(
        X=Y,
        nb_filters=nb_filters,
        compression=compression,
    )

    Y, nb_filters = dense_block(
        X=Y,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=24,
    )

    Y, nb_filters = transition_layer(
        X=Y,
        nb_filters=nb_filters,
        compression=compression,
    )

    Y, nb_filters = dense_block(
        X=Y,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=16,
    )

    Y = keras.layers.AveragePooling2D(
        pool_size=7,
        padding="valid",
    )(Y)

    Y = keras.layers.Dense(
        units=1000,
        activation="softmax",
        kernel_initializer=initializer,
    )(Y)

    return keras.models.Model(inputs=X, outputs=Y)
