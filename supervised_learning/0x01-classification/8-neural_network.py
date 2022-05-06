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

    W1 = 0
    b1 = 0
    A1 = 0
    W2 = 0
    b2 = 0
    A2 = 0

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

        self.W1 = np.random.randn(nodes, nx)
        self.W2 = np.random.randn(1, nodes)
        self.b1 = np.zeros((nodes, 1))
