#!/usr/bin/env python3
"""
    Derivative of Polynomail Calculation Module
"""


from numpy import polymul


def poly_derivative(poly: list):
    """
        Calculates the derivative of a polynomial.

        Args:
            poly (list): List of coefficients representing a polynomial.

        Returns:
            list: A New list of coefficients representing the derivative,
            else it returns [0].
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    result = []

    for index in range(1, len(poly)):

        if not isinstance(index, (int, float)):
            return None

        result.append(index * poly[index])

    return result
