#!/usr/bin/env python3
"""
    Probability binomial representation module.
"""


class Binomial:
    """
        Probability binomial representation class.
    """

    n = None
    p = None

    def __init__(self, data=None, n=1, p=0.5):
        """
            Constructor method.

            Args:
                data (list): list of integers representing the data set.
                n (int): number of trials.
                p (float): probability of success.
        """

        # If data is given
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum(
                [(values - mean) ** 2 for values in data]
            ) / len(data)
            # probability of fail.
            q = 1 - variance / mean
            self.n = round(mean / q)
            self.p = mean / self.n

        # if data is not given
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")

            if p <= 0 or p >= 1:
                raise ValueError(
                    "p must be greater than 0 and less than 1"
                )

            self.n = int(n)
            self.p = float(p)

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

        return self.factorial(self.n) /\
            (self.factorial(k) * self.factorial(self.n - k)) * \
            self.p ** k * (1 - self.p) ** (self.n - k)

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
            cdf += self.pmf(i)

        return cdf

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
