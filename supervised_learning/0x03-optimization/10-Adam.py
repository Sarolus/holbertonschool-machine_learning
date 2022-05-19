#!/usr/bin/env python3
"""
    Tensorflow Adam Optimizer Module
"""

import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
        Creates the operation to perform the Adam optimization
    """

    return tf.train.AdamOptimizer(learning_rate=alpha,
                                  beta1=beta1,
                                  beta2=beta2,
                                  epsilon=epsilon
                                  ).minimize(loss)
