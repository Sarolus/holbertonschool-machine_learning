#!/usr/bin/env python3
"""
    Matrix Inverse Calculation Module
"""
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
        Calculates the inverse matrix.

        Args:
            matrix (list): The given matrix.

        Returns:
            list: The inverse matrix.
    """
    det = determinant(
        matrix,
        value_error_msg='matrix must be a non-empty square matrix'
    )
    adj = adjugate(matrix)

    matrix_length = len(adj)

    if det == 0:
        return None

    inv = [
        [adj[row][column] / det for column in range(matrix_length)]
        for row in range(matrix_length)
    ]

    return inv
