#!/usr/bin/env python3
"""
    Precision Calculation Module
"""

import numpy as np


def precision(confusion):
    """
        Calculates the precision for each class in a confusion matrix

        Args:
            confusion (np.ndarray): Confusion np.ndarray where row indices
            represent the correct labels and column indices represent the
            predicted labels.

        Returns:
            np.ndarray: np.ndarray containing the precision of each class.
    """
    true_positives = np.diagonal(confusion)
    false_positives = np.sum(confusion, axis=0)

    return true_positives / false_positives
