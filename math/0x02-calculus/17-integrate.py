#!/usr/bin/env python3
"""
    Integral of Polynomial Calculation Module
"""


def poly_integral(poly: list, C: int = 0):
    """
        Calculates the integral of a polynomial.

        Args:
            poly (list): List of coefficients representing a polynomial.
            C (int, optional): The constant. Defaults to 0.

        Returns:
            list: The integral of a polynomial.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    integrate = [C]

    for i, coeff in enumerate(poly):
        formula = coeff / (i + 1)
        if formula % 1 == 0:
            integrate.append(int(formula))
        else:
            integrate.append(formula)

    return integrate
