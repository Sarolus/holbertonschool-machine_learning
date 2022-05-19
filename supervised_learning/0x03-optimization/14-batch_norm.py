#!/usr/bin/env python3
"""
    Tensorflow Batch Normalization Module
"""

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
        Creates a batch normalization layer for a neural network in
        tensorflow.

        Args:
            prev (tensor): The activated output of the previous layer.
            n (int): The number of nodes in the layere to be created.
            activation: The activation function that should be used
            on the output of the layer.

        Returns:
            tensor: A tensor of the activated output for the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=initializer)

    mean, variance = tf.nn.moments(layer(prev), axes=[0])

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    batch_norm = tf.nn.batch_normalization(layer(prev),
                                           mean=mean,
                                           variance=variance,
                                           offset=beta,
                                           scale=gamma,
                                           variance_epsilon=1e-8)

    return activation(batch_norm)
