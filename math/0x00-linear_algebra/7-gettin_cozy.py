#!/usr/bin/env python3
"""
    2D Matrix Concatenation Task
"""


def cat_matrices2D(mat1:list, mat2:list, axis:int=0):
    """
        Return the concatenation of two matrix.

        Args:
            arr1 (list): The first matrix.
            arr2 (list): The second matrix.
            axis (int, optional): The specified axis. Defaults to 0.
    """

    # Vertical
    if (axis == 0):
        if len(mat1[0]) != len(mat2[0]):
            return None
        concatenation = [row[:] for row in mat1]
        concatenation += mat2
    
    #Horizontal
    elif (axis == 1):
        if len(mat1) != len(mat2):
            return None

        concatenation = []

        for row in range(len(mat1)):
            concatenation.append(mat1[row] + mat2[row])

    return concatenation