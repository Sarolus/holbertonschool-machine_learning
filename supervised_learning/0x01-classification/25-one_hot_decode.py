#!/usr/bin/env python3
"""
    One-hot Decoding Module
"""
import numpy as np


def one_hot_decode(one_hot):
    """
        Converts a one-hot matrix into a vector of labels

        Args:
            one_hot (np.ndarray): one-hot encoded np.ndarray with shape
            (classes, m).

        Returns:
            np.ndarray: np.ndarray with shape (m,) containing the numeric
            lagbels for each example, or None on failure.
    """
    if type(one_hot) is not np.ndarray:
        return None

    if len(one_hot.shape) != 2:
        return None

    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
