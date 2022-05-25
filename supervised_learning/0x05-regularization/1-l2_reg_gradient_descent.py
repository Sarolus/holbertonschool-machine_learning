#!/usr/bin/env python3
"""
    Performs gradient descent on L2 regularized cost
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        Performs gradient descent on L2 regularized cost

        Args:
            Y: one-hot numpy.ndarray of shape (classes, m) that contains the
                correct labels for the input data
            weights: dictionary of the weights and biases of the neural network
            cache: dictionary of the outputs and inputs of each layer
            alpha: float of the learning rate
            lambtha: regularization parameter
            L: number of layers in the neural network

        Returns:
            updates: dictionary of the weights and biases of the neural network
    """

    m = Y.shape[1]

    for layer_index in range(L, 0, -1):
        A = cache["A" + str(layer_index)]

        if layer_index == L:
            dz = A - Y
        else:
            dz = np.multiply(
                np.dot(weights["W" + str(layer_index + 1)].T, dz),
                A * (1 - A),
            )

        dw = 1 / m * np.dot(dz, cache["A" + str(layer_index - 1)].T) + \
            lambtha / m * weights["W" + str(layer_index)]
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)

        weights["W" + str(layer_index)] = weights[
            "W" + str(layer_index)] - alpha * dw
        weights["b" + str(layer_index)] = weights[
            "b" + str(layer_index)] - alpha * db
