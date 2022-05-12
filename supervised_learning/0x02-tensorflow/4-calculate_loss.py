#!/usr/bin/env python3
"""
    Tensorflow Loss Caculation Module
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
        Calculates the softmax cross-entropy loss
        of a prediction.

        Args:
            y (placeholder): Placeholder for the labels of the input data.
            y_pred (tensor): Tensor containing the network's predictions.

        Returns:
            tensor: Returns a tensor containing the loss of the prediction.
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
