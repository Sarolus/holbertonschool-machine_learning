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

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network.

            Args:
                X (numpy.ndarray): numpy.ndarray whit shape (nx, m) that
                contains the input data.

            Returns:
                int: Returns the Activated Ouputs.
        """

        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

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

    def evaluate(self, X, Y):
        """
            Evaluates the neural network's predictions.

            Args:
                X (np.ndarray): np.ndarray that contains the input features
                to the neuron.
                Y (np.ndarray): np.ndarray that contains the correct labels
                for the input data.

            Returns:
                np.ndarray: Returns the neuron's prediction and the cost of
                the network, respectively.
        """

        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)

        return np.where(A2 >= 0.5, 1, 0), cost
