#!/usr/bin/env python3
"""
    SIGMA Notation for Sum Module
"""


def summation_i_squared(n: int):
    """
        Calculates the SIGMA notation and return the integer value of the sum.

        Args:
            n (int): The stopping condition

        Returns:
            int: The integer value of the sum.
    """
    if n <= 0 or type(n) is not int:
        return None

    if n == 1:
        return 1

    return n ** 2 + summation_i_squared(n - 1)
