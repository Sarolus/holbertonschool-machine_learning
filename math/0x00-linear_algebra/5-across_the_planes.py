#!/usr/bin/env python3
"""
    2D Matrix Addition Task
"""


def add_matrices2D(mat1:list, mat2:list):
    """
        Return the addition of two 2D matrix.

        Args:
            arr1 (list): The first 2D matrix.
            arr2 (list): The second 2D matrix.
    """
    result = []

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    for col_element in range(len(mat1)):
        vector = []
        for row_element in range(len(mat1[0])):
            vector.append(mat1[col_element][row_element] +
                          mat2[col_element][row_element])
        result.append(vector)

    return result
