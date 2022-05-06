#!/usr/bin/env python3
"""_
    Binary Classification Neural Network model.
"""
import numpy as np


class NeuralNetwork:
    """
        Class that defines a neural network with one hidden layer
        performing binary classification.

        Attributes:
            W (int) : The Weights vector for the neuron.
            b (int) : The bias for the neuron.
            A (int) : The activated output of the neuron (prediction).
    """

    __W1 = 0
    __b1 = 0
    __A1 = 0
    __W2 = 0
    __b2 = 0
    __A2 = 0

    def __init__(self, nx: int, nodes: int):
        """
            Initialize the neural network class attributes.

            Args:
                nx (int): The number of input features.
                nodes (int): The number of nodes found in the hidden
                layer.
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__W2 = np.random.randn(1, nodes)
        self.__b1 = np.zeros((nodes, 1))

    @property
    def W1(self):
        """
            Setter method for the Weights vector of the hidden layer.

            Returns:
                int: Returns the Weights Vector of the hidden layer.
        """

        return self.__W1

    @property
    def b1(self):
        """
            Setter method for the Bias of the hidden layer.

            Returns:
                int: Returns the Bias of the hidden layer.
        """

        return self.__b1

    @property
    def A1(self):
        """
            Setter method for the Activated Output of the hidden layer.

            Returns:
                int: Returns the Activated Output of the hidden layer.
        """

        return self.__A1

    @property
    def W2(self):
        """
            Setter method for the Weights vector of the output neuron.

            Returns:
                int: Returns the Weights Vector of the output neuron.
        """

        return self.__W2

    @property
    def b2(self):
        """
            Setter method for the Bias of the output neuron.

            Returns:
                int: Returns the Bias of the output neuron.
        """

        return self.__b2

    @property
    def A2(self):
        """
            Setter method for the Activated Output of the output neuron.

            Returns:
                int: Returns the Activated Output of the output neuron.
        """

        return self.__A2
