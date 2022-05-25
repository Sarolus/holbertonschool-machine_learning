#!/usr/bin/env python3
"""
    Tensorflow Layer Initialize Regulizer Module
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
        Creates a tensorflow layer that includes L2 regularization.

        Args:
            prev (tensor): A tensor containing the output of the
            previous layer.
            n (int): The number of nodes the new layer should
            contain.
            activation (str): The activation function that sould be
            used on the layer.
            lambtha (int): The L2 regularization parameter.

        Returns:
            tensor: The output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode=("fan_avg")
                                                        )
    L2_regularizer = tf.keras.regularizers.L2
    layer = tf.layers.Dense(units=n,
                            kernel_initializer=initializer,
                            kernel_regularizer=L2_regularizer(lambtha),
                            activation=activation
                            )

    return layer(prev)
