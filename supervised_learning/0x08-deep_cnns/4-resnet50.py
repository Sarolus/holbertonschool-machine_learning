#!/usr/bin/env python3
"""
    ResNet-50 Architecture Module
"""
import tensorflow.keras as keras
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
        Builds the ResNet-50 Architecture.

        Returns:
            The keras model.
    """
    X = keras.Input(shape=(224, 224, 3))
    initializer = keras.initializers.HeNormal()

    Y = keras.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=initializer
    )(X)

    batch_norm = keras.layers.BatchNormalization()(Y)
    activation = keras.layers.Activation("relu")(batch_norm)

    Y = keras.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same"
    )(activation)

    Y = projection_block(Y, [64, 64, 256], 1)

    for _ in range(2):
        Y = identity_block(Y, [64, 64, 256])

    Y = projection_block(Y, [128, 128, 512])

    for _ in range(3):
        Y = identity_block(Y, [128, 128, 512])

    Y = projection_block(Y, [256, 256, 1024])

    for _ in range(5):
        Y = identity_block(Y, [256, 256, 1024])

    Y = projection_block(Y, [512, 512, 2048])

    for _ in range(2):
        Y = identity_block(Y, [512, 512, 2048])

    Y = keras.layers.AveragePooling2D(
        pool_size=7,
        padding="same",
    )(Y)

    Y = keras.layers.Dense(
        units=1000,
        activation="softmax",
        kernel_initializer=initializer
    )(Y)

    return keras.models.Model(inputs=X, outputs=Y)
