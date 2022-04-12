#!/usr/bin/env python3
"""
    Matrix Numpy Concatenation Task
"""


import numpy as np


def np_cat(mat1: list, mat2: list, axis: int = 0):
    """
        Return the concatenation between the first and the second specified matrix.

        Args:
            mat1 (list): The first specified matrix.
            mat2 (list): The second specified matrix.
            axis (int, optional): The specified axis. Defaults to 0.

        Returns:
        _   list: The concatenation of the first and second matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
