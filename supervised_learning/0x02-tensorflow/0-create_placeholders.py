#!/usr/bin/env python3
"""
    Tensorflow Placeholders Initialization Module
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for the neural network.

    Args:
        nx (int): The number of feature columns in our data.
        classes (int): The number of classes in our classifier.

    Returns:
        tf.placeholder: Returns the placeholders named x and y,
        respectively.
    """

    # The placeholder for the input data to the neural network.
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    # The placeholder for the one-hot labels for the input data.
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
