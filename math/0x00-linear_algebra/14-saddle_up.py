#!/usr/bin/env python3
"""
    Matrix Numpy Multiplication Task
"""

import numpy as np


def np_matmul(mat1: list, mat2: list):
    """
        Return the result of the multiplication between the first and the second matrix.

        Args:
            mat1 (list): The first specified matrix.
            mat2 (list): The second specified matrix.

        Returns:
            list: Result of the multiplication of the first and the second matrix.
    """
    result = np.matmul(mat1, mat2)
    return result
