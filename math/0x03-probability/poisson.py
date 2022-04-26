#!/usr/bin/env python3
"""
    Probability Poisson Distribution Representation
"""


class Poisson:
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

            self.lambtha = float(sum(data) / len(data))

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

            self.lambtha = float(lambtha)

    def pmf(self, k):
        """
            Summary

            Args:
                k (_type_): _description_

            Returns:
                _type_: _description_
        """
        if type(k) is not int:
            k = int(k)

        if k < 0:
            return 0

        euler = 2.7182818285

        return (self.lambtha ** k * euler ** (-self.lambtha)) /\
            self.factorial(k)

    def cdf(self, k):
        """
            Summary

            Args:
                k (_type_): _description_

            Returns:
                _type_: _description_
        """
        if type(k) is not int:
            k = int(k)

        if k < 0:
            return 0

        cdf = 0

        for i in range(k + 1):
            cdf += self.pnf(i)

    def factorial(self, n):
        """
            Summary

            Args:
                n (_type_): _description_

            Returns:
                _type_: _description_
        """
        factorial = 1

        for i in range(1, n + 1):
            factorial *= i

        return factorial
