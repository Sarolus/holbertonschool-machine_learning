#!/usr/bin/env python3
"""
    Tensorflow Train Operation Module
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
        Creates the training operation for the network.

        Args:
            loss (tensor): The loss of the network's prediction.
            alpha (int): The learning rate.

        Returns:
            Returns an operation that trains the network using
            gradient descent.
    """

    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
