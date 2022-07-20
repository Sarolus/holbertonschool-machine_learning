#!/usr/bin/env python3
"""
    Matrix Determinant Calculation Module
"""


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


def determinant(matrix,
                type_error_msg='matrix must be a list of lists',
                value_error_msg='matrix must be a square matrix'
                ):
    """
        Calculates the determinant of the given matrix.

        Args:
            matrix (list): The given matrix.

        Returns:
            int: The determinant of the given matrix.
    """
    matrix_length = len(matrix)
    result = 0

    if not matrix or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    for row in matrix:
        if type(row) is not list:
            raise TypeError(type_error_msg)

        if len(row) != matrix_length:
            raise ValueError(value_error_msg)

    if matrix_length == 1:
        return matrix[0][0]

    for row_index in range(matrix_length):
        value = matrix[0][row_index]
        sub_matrix = create_sub_matrix(matrix_length, matrix, row_index)
        sub_determinant = determinant(sub_matrix)
        result += ((-1) ** row_index) * value * sub_determinant

    return result
