#!/usr/bin/env python3
"""
    Inception Network Module
"""
import tensorflow.keras as keras
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
        Builds the inception network.

        Returns:
            The keras model.
    """
    X = keras.Input(shape=(224, 224, 3))

    Y = keras.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        activation="relu"
    )(X)

    Y = keras.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same"
    )(Y)

    Y = keras.layers.Conv2D(
        filters=192,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(Y)

    Y = keras.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same"
    )(Y)

    Y_inception = inception_block(Y, (64, 96, 128, 16, 32, 32))
    Y_inception = inception_block(Y_inception, (128, 128, 192, 32, 96, 64))

    Y = keras.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same"
    )(Y_inception)

    Y_inception = inception_block(Y, (192, 96, 208, 16, 48, 64))
    Y_inception = inception_block(Y_inception, (160, 112, 224, 24, 64, 64))
    Y_inception = inception_block(Y_inception, (128, 128, 256, 24, 64, 64))
    Y_inception = inception_block(Y_inception, (112, 144, 288, 32, 64, 64))
    Y_inception = inception_block(Y_inception, (256, 160, 320, 32, 128, 128))

    Y = keras.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same"
    )(Y_inception)

    Y_inception = inception_block(Y, (256, 160, 320, 32, 128, 128))
    Y_inception = inception_block(Y_inception, (384, 192, 384, 48, 128, 128))

    Y = keras.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding="valid"
    )(Y_inception)

    dropout = keras.layers.Dropout(0, 4)(Y)

    Y = keras.layers.Dense(
        units=1000,
        activation="softmax"
    )(dropout)

    model = keras.models.Model(inputs=X, outputs=Y)

    return model
