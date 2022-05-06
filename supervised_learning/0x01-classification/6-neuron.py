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

    def evaluate(self, X, Y):
        """
            Evaluates the neuron's predictions.

            Args:
                X (np.ndarray): np.ndarray that contains the input data.
                Y (np.ndarray): np.ndarray that contains the correct labels
                for the input data.

            Returns:
                np.ndarray: Returns the neuron's prediction and the cost of
                the network, respectively.
        """

        A = self.forward_prop(X)
        cost = self.cost(Y, A)

        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(
        self, X: np.ndarray,
        Y: np.ndarray,
        A: np.ndarray,
        alpha: float = 0.05
    ):
        """
            Caculates one pass of gradient descent on the neuron.

            Args:
                X (np.ndarray): np.ndarray that contains the input data.
                Y (np.ndarray): np.ndarray that contains the correct labels for
                the input data.
                A (np.ndarray): np.ndarray that contains the activated output
                of the neuron for each example.
                alpha (float, optional): The learning rate of the neuron.
                Defaults to 0.05.
        """
        z_derivative = A - Y
        W_derivative = 1 / X.shape[1] * np.dot(X, z_derivative.T)
        b_derivative = 1 / X.shape[1] * np.sum(z_derivative)
        self.__W -= alpha * W_derivative.T
        self.__b -= alpha * b_derivative

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
                over.
                Defaults to 5000.
                alpha (float, optional): The learning rate of the neuron.
                Defaults to 0.05.

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
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

        return self.evaluate(X, Y)
