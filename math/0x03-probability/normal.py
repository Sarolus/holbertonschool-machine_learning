#!/usr/bin/env python3
"""
    Probability Normal Distribution Representation
"""


class Normal:
    """
        Normal Distribution Methods
    """

    def __init__(self, data=None, mean=0., stddev=1.):
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

            self.mean = float(sum(data) / len(data))
            self.stddev = pow((sum(pow(x - self.mean, 2)
                              for x in data) / len(data)), 0.5)

        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """
            Summary

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_
        """

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
            Summary

            Args:
                z (_type_): _description_

            Returns:
                _type_: _description_
        """

        return self.mean + z * self.stddev

    def pdf(self, x):
        """
            Summary

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_
        """

        e = 2.7182818285
        pi = 3.1415926536

        return e ** (-0.5 * self.z_score(x) ** 2) /\
            (self.stddev * (2 * pi) ** 0.5)

    def cdf(self, x):
        """
            Cumulative distribution function.
        """

        return 0.5 * (1 + self._erf((x - self.mean) /
                                    (self.stddev * pow(2, 0.5))))

    def _erf(self, x):
        """
            Error function approximation
        """

        pi = 3.1415926536

        return 2 / pow(pi, 0.5) *\
            (x - pow(x, 3)/3 + pow(x, 5)/10 -
                pow(x, 7)/42 + pow(x, 9)/216)
