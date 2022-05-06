#!/usr/bin/env python3
"""_
    Binary Classification Single Neuron model.
"""
import numpy as np


class Neuron:
    """
        Class that defines a single neuron perfoming binary classification.

        Attributes:
            W (int) : The Weights vector for the neuron.
            b (int) : The bias for the neuron.
            A (int) : The activated output of the neuron (prediction).
    """

    W = 0
    b = 0
    A = 0

    def __init__(self, nx: int):
        """
            Initialize the neuron class attributes.

            Args:
                nx (int): The numvber of input features to the neuron.

            Raises:
                TypeError: nx must be of type int.
                ValueError: nx cannot be less than 1.
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
