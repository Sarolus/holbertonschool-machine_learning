#!/usr/bin/env python3
"""
    Concatenates two matrices along a specific axis
"""


def matrix_shape(matrix):
    """
        Returns the shape of a matrix
    """

    shape = []

    while(type(matrix) is list):
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape


def cat_matrices(mat1, mat2, axis=0):
    """
        Concatenates two matrices along a specific axis
    """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)

    if len(shape1) != len(shape2):
        return None

    for row in range(len(shape1)):
        if shape1[row] != shape2[row] and row != axis:
            return None

    if axis < 0 or axis > len(shape1):
        return None

    result = []

    if axis == 0:
        return mat1 + mat2

    for row in range(len(mat1)):
        result.append(cat_matrices(mat1[row], mat2[row], axis - 1))

    return result
