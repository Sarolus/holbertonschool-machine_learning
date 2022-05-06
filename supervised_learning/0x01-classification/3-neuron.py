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

    __W = 0
    __b = 0
    __A = 0

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

        self.__W = np.random.randn(1, nx)

    @property
    def W(self):
        """
            Setter method for the Weights vector.

            Returns:
                int: Returns the Weights Vector.
        """

        return self.__W

    @property
    def b(self):
        """
            Setter method for the Bias.

            Returns:
                int: Returns the Bias.
        """

        return self.__b

    @property
    def A(self):
        """
            Setter method for the Activated Output.

            Returns:
                int: Returns the Activated Output.
        """

        return self.__A

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neuron.

            Args:
                X (np.ndarray): np.ndarray with shape (nx, m) that
                contains the input data.

            Returns:
                int: Returns the Activated Ouput.
        """

        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (np.ndarray): np.ndarray that contains the correct labels
            for the input data.
            A (np.ndarray): np.ndarray that contains the activated output
            of the neuron for each example.

        Returns:
            int: Returns the cost of the model.
        """

        observations = Y.shape[1]
        precision = 1.0000001

        # Take the error when Y = 1.
        class1_cost = -Y * np.log(A)

        # Take the error when Y = 0.
        class2_cost = (1 - Y) * np.log(precision - A)

        # Take the sum of both costs.
        cost = class1_cost - class2_cost

        # Take the average cost.
        cost = cost.sum() / observations

        return cost
