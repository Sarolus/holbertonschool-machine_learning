#!/usr/bin/env python3
"""
    One-hot encoding.
"""

import tensorflow.keras as keras


def one_hot(labels, classes=None):
    """
        One-hot encoding

        Args:
            labels: list of class labels
            classes: number of classes

        Returns:
            one-hot encoded labels
    """

    return keras.utils.to_categorical(labels, num_classes=classes)
