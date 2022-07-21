#!/usr/bin/env python3
"""
    Matrix Cofactor Calculation Module
"""

minor = __import__('1-minor').minor


def cofactor(matrix):
    """
        Calculates the cofactor matrix.

        Args:
            matrix (list): The given matrix.

        Returns:
            list: The cofactor matrix.
    """
    cofactor = minor(matrix)
    matrix_length = len(matrix)

    for row in range(matrix_length):
        for column in range(matrix_length):
            if (row + column) % 2 != 0:
                cofactor[row][column] *= -1 ** (row + column)

    return cofactor
