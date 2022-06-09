#!/usr/bin/env python3
"""
    LeNet-5 Architecture with Keras Module
"""
import tensorflow.keras as keras


def lenet5(X):
    """
        Modified version of the LeNet-5 architecture using
        Keras.

        Args:
            x (tf.placeholder): Contains the input images.
            y (tf.placeholder): Contains the one-hot labels.

        Returns:
            tensor: a tensor for the softmax activated output.
            tensor: training operation that utilizes Adam optim.
            tensor: a tensor for the loss of the network.
            tensor: a tensor for the accuracy of the network.
    """
    initializer = keras.initializers.HeNormal()

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        kernel_initializer=initializer
    ))

    model.add(keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(keras.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation="relu",
        kernel_initializer=initializer
    ))

    model.add(keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(
        units=120,
        activation="relu",
        kernel_initializer=initializer
    ))

    model.add(keras.layers.Dense(
        units=84,
        activation="relu",
        kernel_initializer=initializer
    ))

    model.add(keras.layers.Dense(
        units=10,
        activation="softmax",
        kernel_initializer=initializer
    ))

    adam = keras.optimizers.Adam()

    model.compile(optimizer=adam,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"]
                  )

    return model
