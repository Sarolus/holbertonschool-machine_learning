#!/usr/bin/env python3
"""
    Creates the operation to perform the batch normalization
"""

import tensorflow.compat.v1 as tf


def batch_norm(Z, gamma, beta, epsilon):
    """
        Creates the operation to perform the batch normalization
    """

    mean, var = tf.nn.moments(tf.cast(Z, tf.float16), axes=[0])

    return tf.nn.batch_normalization(Z, mean, var, beta, gamma, epsilon)
