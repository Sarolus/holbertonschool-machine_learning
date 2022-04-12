#!/usr/bin/env python3
"""
    Matrix Multiplication Task
"""

from unittest import result


def mat_mul(mat1: list, mat2: list):
    """
        Make the multiplication between 

    Args:
        mat1 (list): _description_
        mat2 (list): _description_

    Returns:
        list: Return the result of the multiplication
    """
    result = []

    for row in range(len(mat1)):
        result.append([])

        for column in range(len(mat2[0])):
            # Initialize the result matrix.
            result[row].append(0)

            for element in range(len(mat2)):
                # Multiply the row of mat1 with the column of mat2.
                result[row][column] += mat1[row][element] * \
                    mat2[element][column]

    return result
