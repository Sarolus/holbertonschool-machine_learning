#!/usr/bin/env python3
"""
    Tensorflow Layer Creation Module
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
        Returns the tensor output of the layer.

        Args:
            prev: The tensor output of the previous layer.
            n (int): The number of nodes in the layer to create.
            activation (string): The activation function that
            the layer should use.

        Returns:
            Returns the tensor output of the layer.
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name="layer")

    return layer(prev)
