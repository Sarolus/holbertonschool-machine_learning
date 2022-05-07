#!/usr/bin/env python3
"""
    One-hot Encoding Module
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
        Converts a numeric label vector into a one-hot matrix

        Args:
            Y (np.ndarray): np.ndarray containing numeric class labels.
            classes (int): The maximum number of classes found in Y.

        Returns:
            np.ndarray: A one-hot encoding of Y wit shape (classes, m),
            or None on failure.
    """
    if type(Y) is not np.ndarray:
        return None

    if type(classes) is not int:
        return None

    try:
        return np.eye(classes)[Y].T
    except Exception:
        return None
