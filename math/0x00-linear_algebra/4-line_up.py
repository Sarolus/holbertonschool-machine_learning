#!/usr/bin/env python3
"""
    Matrix Addition Task
"""


def add_arrays(arr1: list, arr2: list):
    """
        Return the addition of two matrix.

        Args:
            arr1 (list): The first matrix.
            arr2 (list): The second matrix.
    """
    result = []

    if len(arr1) != len(arr2):
        return None

    for element in range(len(arr1)):
        result.append(arr1[element] + arr2[element])

    return result
