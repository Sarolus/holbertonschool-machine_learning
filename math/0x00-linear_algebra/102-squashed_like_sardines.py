#!/usr/bin/env python3
"""
    Matrix Concatenation Based On Specified Axis Task
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


def cat_matrices(mat1: list, mat2: list, axis: int = 0):
    """
        Return the concatenation between the first and the second
        specified matrix.

        Args:
            mat1 (list): The first specified matrix.
            mat2 (list): The second specified matrix.
            axis (int, optional): The specified axis. Defaults to 0.

        Returns:
        _   list: The concatenation of the first and second matrix.
    """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)

    if len(shape1) != len(shape2):
        return None

    # Y axis
    if axis == 0:
        return mat1 + mat2

    # X axis
    elif axis == 1:
        result = []

        for row in range(len(mat1)):
            result.append(mat1[row] + mat2[row])

        return result

    # N axis
    elif axis == 3:
        result = []

        for row in range(len(mat1)):
            result.append([])
            for column in range(len(mat1[row])):
                result[row].append([])

                for i in range(len(mat1[row][column])):
                    result[row][column].append(
                        mat1[row][column][i] + mat2[row][column][i])

        return result

    return None
