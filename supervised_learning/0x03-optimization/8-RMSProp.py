#!/usr/bin/env python3
"""
    Tensorflow RMS Prop Optimizer Module
"""

import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
        Creates the operation to perform the RMSProp optimization
    """

    return tf.train.RMSPropOptimizer(learning_rate=alpha,
                                     decay=beta2,
                                     epsilon=epsilon
                                     ).minimize(loss)
