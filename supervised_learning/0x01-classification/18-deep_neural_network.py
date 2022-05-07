#!/usr/bin/env python3
"""_
    Binary Classification Deep Neural Network model.
"""
import numpy as np


class DeepNeuralNetwork:
    """
        Class that defines a deep neural network with one hidden layer
        performing binary classification.

        Attributes:
            L (list) : The number of layers in the neural network.
            cache (dict) : A dictionary to hold all intermediary values
            of the network.
            weights (dict) : A dictionary to hold all weights and biased
            of the network.
    """

    __L = []
    __cache = {}
    __weights = {}

    def __init__(self, nx: int, layers: list):
        """
            Initialize the deep neural network class attributes.

            Args:
                nx (int): The number of input features.
                layers (list): The list representing the number of nodes
                in each layer of the network.
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")

        if any(list(map(lambda x: x <= 0, layers))):
            raise TypeError("layers must be a list of positive integers")

        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_nb in range(self.__L):
            current_layer = layers[layer_nb]

            if layer_nb == 0:
                self.__weights["W" + str(layer_nb + 1)] = np.random.randn(
                    current_layer, nx
                ) * np.sqrt(2 / nx)

            else:
                previous_layer = layers[layer_nb - 1]
                self.__weights["W" + str(layer_nb + 1)] = np.random.randn(
                    current_layer, previous_layer
                ) * np.sqrt(2 / previous_layer)

            self.__weights["b" + str(layer_nb + 1)] = np.zeros(
                (current_layer, 1)
            )

    @property
    def L(self):
        """
            Setter method for the number of layers in the neural network.

            Returns:
                list: Returns the number of layers in the neural network.
        """

        return self.__L

    @property
    def cache(self):
        """
            Setter method for the dictionary to hold all intermediary values
            of the network.

            Returns:
                dict: Returns the dictionary to hold all intermediary values
                of the network.
        """

        return self.__cache

    @property
    def weights(self):
        """
            Setter method for the dictionary to hold all weights and biased
            of the network.

            Returns:
                dict: Returns the dictionary to hold all weights and biased
                of the network.
        """

        return self.__weights

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the deep neural network.

            Args:
                X (numpy.ndarray): numpy.ndarray whit shape (nx, m) that
                contains the input data.

            Returns:
                int: Returns the Activated Ouputs of the neural network
                and the cache.
        """

        self.__cache["A0"] = X

        for layer_index in range(self.__L):
            a_previous = self.__cache["A" + str(layer_index)]
            w = self.__weights["W" + str(layer_index + 1)]
            b = self.__weights["b" + str(layer_index + 1)]
            Z = np.dot(w, a_previous) + b
            self.__cache["A" + str(layer_index + 1)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A" + str(self.__L)], self.__cache
