#!/usr/bin/env python3
"""
    Tensorflow Accuracy Calculation Module
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
        Calculates the accuracy of a prediction.

        Args:
            y (placeholder): Placeholder for the labels of the input data.
            y_pred (tensor): Tensor containing the network's predictions.

        Returns:
            tensor: Returns a tensor containing the decimal accuracy of
            the prediction.
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
