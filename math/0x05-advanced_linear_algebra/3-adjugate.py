#!/usr/bin/env python3
"""
    Matrix Adjugate Calculation Module
"""

cofactor = __import__('2-cofactor').cofactor


def matrix_transpose(matrix):
    """
        Returns the transpose of a matrix
    """

    if type(matrix) is not list:
        return None

    if type(matrix[0]) is not list:
        return None

    if len(matrix) == 0:
        return None

    if len(matrix[0]) == 0:
        return None

    transpose = []

    for row in range(len(matrix[0])):
        transpose.append([])
        for column in range(len(matrix)):
            transpose[row].append(matrix[column][row])

    return transpose


def adjugate(matrix):
    """
        Calculates the adjugate matrix.

        Args:
            matrix (list): The given matrix.

        Returns:
            list: The adjugate matrix.
    """
    cofactor_matrix = cofactor(matrix)

    return matrix_transpose(cofactor_matrix)
