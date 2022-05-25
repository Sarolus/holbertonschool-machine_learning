#!/usr/bin/env python3
"""
    Tensorflow Regularization Cost Calculation Module
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
        Calculates the cost of a neural network with L2 regularization.

        Args:
            cost (tensor): A tensor containing the cost of the network
            without L2 regularization.

        Returns:
            tensor: A tensor containing the cost of the network
            accounting for L2 regularization.
    """
    return cost + tf.losses.get_regularization_losses()
