#!/usr/bin/env python3
"""
    Matrix Multi-Dimensional Additions Task
"""


def matrix_shape(matrix: list):
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


def add_matrices(mat1: list, mat2: list):
    """
        Return the addition of two matrix between each other.

        Args:
            mat1 (list): The first specified matrix.
            mat2 (list): The second specified matrix.

        Returns:
        _   list: Result of the addition.
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    result = []

    for row in range(len(mat1)):
        result.append([])
        if type(mat1[row]) is list:
            for col in range(len(mat1[row])):
                if type(mat1[row][col]) is list:
                    result[row].append(add_matrices(
                        mat1[row][col], mat2[row][col]))
                else:
                    result[row].append(mat1[row][col] + mat2[row][col])
        else:
            result[row] = mat1[row] + mat2[row]

    return result
