#!/usr/bin/env python3
"""
    Specificity Calculation Module
"""

import numpy as np


def specificity(confusion):
    """
        Calculates the specify for each class in a confusion matrix.

        Args:
            confusion (np.ndarray): Confusion np.ndarray where row indices
            represent the correct labels and column indices represent the
            predicted labels.

        Returns:
            np.ndarray: np.ndarray containing the specificity of each class.
    """

    true_positives = np.diagonal(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    true_negatives = np.sum(confusion) - false_positives - \
        false_negatives - true_positives

    return true_negatives / (true_negatives + false_positives)
