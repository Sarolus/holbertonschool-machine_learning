#!/usr/bin/env python3
"""
    Confusion Matrix Initializer Module
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
        Creates a confusion matrix.

        Args:
            labels (np.ndarray): One-hot np.ndarray containing the correct
            labels for each data point.
            logits (np.ndarray): One-hot np.ndarray containing the predicted
            labels.

        Returns:
            np.ndarray: A confusion np.ndarray with row indices representing
            the correct labels and column indices representing the predicted
            labels.
    """
    confusion_matrix = np.zeros((labels.shape[1], logits.shape[1]))

    for index in range(labels.shape[0]):
        confusion_matrix[
            np.argmax(labels[index])
        ][
            np.argmax(logits[index])
        ] += 1

    return confusion_matrix
