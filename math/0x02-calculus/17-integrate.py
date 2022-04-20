#!/usr/bin/env python3
"""
    Integral of Polynomial Calculation Module
"""


def poly_integral(poly, C=0):
    """
        Calculates the integral of a polynomial.

        Args:
            poly (list): List of coefficients representing a polynomial.
            C (int, optional): The constant. Defaults to 0.

        Returns:
            list: The integral of a polynomial.
    """
    if (
        type(poly) is not list or
        not isinstance(C, (int, float))
        or len(poly) == 0
    ):
        return None

    if poly == [0]:
        return [C]

    integrate = [C]

    for i, coeff in enumerate(poly):

        if not isinstance(coeff, (int, float)):
            return None

        formula = coeff / (i + 1)
        integrate.append(int(formula) if formula.is_integer() else formula)

    return integrate
