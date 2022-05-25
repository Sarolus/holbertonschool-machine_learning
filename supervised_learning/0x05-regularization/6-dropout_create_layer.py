#!/usr/bin/env python3
"""
    Tensorflow Layer Initialize With Dropout Module
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        Creates a tensorflow layer using dropout

        Args:
            prev (tensor): A tensor containing the output of the
            previous layer.
            n (int): The number of nodes the new layer should
            contain.
            activation (str): The activation function that sould be
            used on the layer.
            keep_prob (int): Probability that a node will be kept.

        Returns:
            tensor: The output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode=("fan_avg")
                                                        )
    drop_out = tf.layers.Dropout(rate=1 - keep_prob)
    layer = tf.layers.Dense(units=n,
                            kernel_initializer=initializer,
                            activation=activation
                            )

    return drop_out(layer(prev))
