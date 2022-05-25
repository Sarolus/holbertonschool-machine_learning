#!/usr/bin/env python3
"""
    Regularization Cost Calculation Module
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
        Calculates the cost of a neural network with L2 regularization.

        Args:
            cost (int): Cost of a neural network.
            lambtha (int): Regularization parameter.
            weights (dict): Dictionary of weights and biases.
            L (int): Number of layers.
            m (int): Number of data points.

        Returns:
            int: The cost of the network accounting for L2 regularization
    """
    for index in range(1, L + 1):
        sum = np.sum(np.square(weights["W" + str(index)]))

    return cost + lambtha * sum / (2 * m)
