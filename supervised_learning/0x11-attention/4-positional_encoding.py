#!/usr/bin/env python3
"""
    Script that calculates the positional encoding for a transformer
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
        Calculates the positional encoding for a transformer

        Args:
            max_seq_len: is an integer representing the maximum sequence
                length
            dm: is the model depth

        Returns:
            a numpy.ndarray of shape (max_seq_len, dm) containing the
                positional encoding vectors
    """

    # Create an array of shape (max_seq_len, 1)
    pos = np.arange(max_seq_len)[:, np.newaxis]

    # Create an array of shape (1, dm)
    i = np.arange(dm)[np.newaxis, :]

    # Calculate the angle
    angle = pos / np.power(10000, (2 * (i // 2)) / np.float32(dm))

    # Apply sin to even indices in the array; 2i
    angle[:, 0::2] = np.sin(angle[:, 0::2])

    # Apply cos to odd indices in the array; 2i + 1
    angle[:, 1::2] = np.cos(angle[:, 1::2])

    return angle
