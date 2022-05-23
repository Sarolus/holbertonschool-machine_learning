#!/usr/bin/env python3
"""
    Sensitivity Calculation Module
"""

import numpy as np


def sensitivity(confusion):
    """
        Calculates the sensitivity for each class in a confusion matrix.

        Args:
            confusion (np.ndarray): Confusion np.ndarray where row
            indices represent the correct labels and column indices
            represent the predicted labels.

        Returns:
            np.ndarray: A np.ndarray containing the sensitivity of each class.
    """
    true_positives = np.diagonal(confusion)
    false_negatives = np.sum(confusion, axis=1)

    return true_positives / false_negatives
