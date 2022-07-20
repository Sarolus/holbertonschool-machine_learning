#!/usr/bin/env python3
"""
    Matrix Minor Calculation Module
"""


determinant = __import__('0-determinant').determinant


def get_matrix_minor(matrix, row, column):
    """
        Get the minor sub matrix.

        Args:
            matrix (list): The given matrix.
            row (int): The row index.
            column (int): The column index.

        Returns:
            list: The minor sub matrix.
    """
    return [
        row[:column] + row[column + 1:]
        for row in (matrix[:row] + matrix[row + 1:])
    ]


def create_sub_matrix(matrix_length, matrix, row_index):
    """
        Create the sub_matrix based on the given matrix.

        Args:
            matrix_length (int): Length of the given matrix.
            matrix (list): The given matrix
            row_index (int): Index of the length of the given matrix.

        Returns:
            list: The sub_matrix needed for the sub_determinant.
    """
    sub_matrix = [
        [
            matrix[row + 1]
            [column if column < row_index else column + 1]
            for column in range(matrix_length - 1)
        ]
        for row in range(matrix_length - 1)
    ]

    return sub_matrix


def minor(matrix):
    """
        Calculates the minor of the given matrix.

        Args:
            matrix (list): The given matrix.

        Returns:
            int: The minor of the given matrix.
    """
    matrix_length = len(matrix)
    result = []

    if not matrix or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')

        if len(row) != matrix_length:
            raise ValueError('matrix must be a non-empty square matrix')

    sub_result = []
    if matrix_length == 1:
        sub_result.append([1])
        return sub_result

    for row_index in range(matrix_length):
        sub_det = []
        for column_index in range(matrix_length):
            sub_matrix = get_matrix_minor(matrix, row_index, column_index)
            sub_determinant = determinant(sub_matrix)
            sub_det.append(sub_determinant)
        result.append(sub_det)

    return result
