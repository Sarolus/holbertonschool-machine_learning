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

    def gradient_descent(
        self, X: np.ndarray,
        Y: np.ndarray,
        A1: np.ndarray,
        A2: np.ndarray,
        alpha: float = 0.05
    ):
        """
            Caculates one pass of gradient descent on the neural network.

            Args:
                X (np.ndarray): np.ndarray that contains the input data.
                Y (np.ndarray): np.ndarray that contains the correct labels for
                the input data.
                A1 (np.ndarray): np.ndarray that contains the activated output
                of the hidden layer.
                A2 (np.ndarray) : np.ndarray that contains the predicted
                output.
                alpha (float, optional): The learning rate of the neuron.
        """
        m = X.shape[1]
        z2_derivative = A2 - Y
        W2_derivative = np.dot(z2_derivative, A1.T) / m
        b2_derivative = np.sum(z2_derivative, axis=1, keepdims=True) / m

        z1_derivative = np.dot(self.__W2.T, z2_derivative) * (A1 * (1 - A1))
        W1_derivative = np.dot(z1_derivative, X.T) / m
        b1_derivative = np.sum(z1_derivative, axis=1, keepdims=True) / m

        self.__W2 -= alpha * W2_derivative
        self.__b2 -= alpha * b2_derivative
        self.__W1 -= alpha * W1_derivative
        self.__b1 -= alpha * b1_derivative

    def train(
        self, X: np.ndarray,
        Y: np.ndarray,
        iterations: int = 5000,
        alpha: float = 0.05
    ):
        """
            Trains the neuron.

            Args:
                X (np.ndarray): np.ndarray that contains the input data.
                Y (np.ndarray): np.ndarray that contains the correct labels for
                the input data.
                iterations (int, optional): The number of iterations to train
                over. Defaults to 5000.
                alpha (float, optional): The learning rate of the neural
                network. Defaults to 0.05.

            Returns:
                np.ndarray: Returns the evaluation of the training data after
                iterations of training have occured.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")

        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")

        if alpha < 0:
            raise ValueError("alpha must be positive")

        for index in range(iterations):
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
