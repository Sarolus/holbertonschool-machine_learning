#!/usr/bin/env python3
"""
    Matrix All Mathematics Operations Task
"""


def np_elementwise(mat1: list, mat2: list):
    """
        Return all mathematics operations of two matrix.

        Args:
            mat1 (list): The first specified matrix.
            mat2 (list): The second specified matrix.

        Returns:
            tuple: Results of all the operations.
    """
    result_add = mat1 + mat2
    result_sub = mat1 - mat2
    result_mul = mat1 * mat2
    result_div = mat1 / mat2

    return (result_add, result_sub, result_mul, result_div)
