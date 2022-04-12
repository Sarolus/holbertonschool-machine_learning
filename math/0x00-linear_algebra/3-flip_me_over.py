#!/usr/bin/env python3
"""
    Matrix Transpose Task
"""


def matrix_transpose(matrix: list):
    """
        Return the transpose of the specified 2D matrix.

        Args:
            matrix (list): The specified 2D matrix
    """
    transpose = []

    for col_element in range(len(matrix[0])):
        vector = []
        for row_element in range(len(matrix)):
            vector.append(matrix[row_element][col_element])
        transpose.append(vector)

    return transpose
