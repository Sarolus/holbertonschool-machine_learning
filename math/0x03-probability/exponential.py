#!/usr/bin/env python3
"""
    Probability Exponential Distribution Representation
"""


class Exponential:
    """
        Summary

        Raises:
            TypeError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
    """

    lambtha = None

    def __init__(self, data=None, lambtha=1.):
        """_
            Summary

            Args:
                data (_type_, optional): _description_. Defaults to None.
                lambtha (_type_, optional): _description_. Defaults to 1..

            Raises:
                TypeError: _description_
                ValueError: _description_
                ValueError: _description_
        """
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(len(data) / sum(data))

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

            self.lambtha = float(lambtha)

    def pdf(self, x):
        """*
            summary

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_
        """
        if not isinstance(x, (int, float)):
            x = float(x)

        if x < 0:
            return 0

        euler = 2.7182818285

        return (self.lambtha * euler ** (-self.lambtha * x))

    def cdf(self, x):
        """
            Summary

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_
        """

        if not isinstance(x, (int, float)):
            x = float(x)

        if x < 0:
            return 0

        euler = 2.7182818285

        return (1 - euler ** (-self.lambtha * x))
