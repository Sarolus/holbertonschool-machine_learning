#!/usr/bin/env python3
"""
    F1 Score Calculation Module
"""

import numpy as np
sensitivity = __import__("1-sensitivity").sensitivity
precision = __import__("2-precision").precision


def f1_score(confusion):
    """
        Calculates the F1 score for each class in a confusion matrix.

        Args:
            confusion (np.ndarray): Confusion np.ndarray where row indices
            represent the correct labels and column indices represent the
            predicted labels.

        Returns:
            np.ndarray: np.ndarray containing the F1 score of each class.
    """

    recall = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * recall) / (prec + recall)
