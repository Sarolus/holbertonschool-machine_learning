#!/usr/bin/env python3
"""
    Matrix Shape Task
"""


def matrix_shape(matrix:list):
    """
        Return the shape of a specified matrix.

        Args:
            matrix (list): The specified matrix

        Returns:
            list: The shape of a matrix
    """
    shape = []

    while(type(matrix) is list):
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape
