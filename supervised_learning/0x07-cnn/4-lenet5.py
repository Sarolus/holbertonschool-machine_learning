#!/usr/bin/env python3
"""
    LeNet-5 Architecture with Tensorflow Module
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
        Modified version of the LeNet-5 architecture using
        tensorflow.

        Args:
            x (tf.placeholder): Contains the input images.
            y (tf.placeholder): Contains the one-hot labels.

        Returns:
            tensor: a tensor for the softmax activated output.
            tensor: training operation that utilizes Adam optim.
            tensor: a tensor for the loss of the network.
            tensor: a tensor for the accuracy of the network.
    """
    initializer = tf.keras.initializers.VarianceScaling()

    conv_layer_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        kernel_initializer=initializer
    )(x)

    max_pool_layer_1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv_layer_1)

    conv_layer_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation="relu",
        kernel_initializer=initializer
    )(max_pool_layer_1)

    max_pool_layer_2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv_layer_2)

    flatten = tf.layers.Flatten()(max_pool_layer_2)

    dense_layer_1 = tf.layers.Dense(
        units=120,
        activation="relu",
        kernel_initializer=initializer
    )(flatten)

    dense_layer_2 = tf.layers.Dense(
        units=84,
        activation="relu",
        kernel_initializer=initializer
    )(dense_layer_1)

    output_layer = tf.layers.Dense(
        units=10,
        kernel_initializer=initializer
    )(dense_layer_2)

    y_prediction = tf.nn.softmax(output_layer)

    loss = tf.losses.softmax_cross_entropy(y, y_prediction)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_prediction, train_op, loss, accuracy
