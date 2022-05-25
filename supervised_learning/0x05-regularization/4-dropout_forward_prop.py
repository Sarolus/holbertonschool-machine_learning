#!/usr/bin/env python3
"""
    Forward Propagation With Dropout Module
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
        Conducts forward propagation using dropout.

        Args:
            X (np.ndarray): np.ndarray containing the input data
            for the network.
            weights (dict): Dictionary of the weights and biases
            of the neural network.
            L (int): Number of layers in the network.
            keep_prob (int): PRobability that a node will be kept.

        Returns:
            dict: Dictionary containing the outputs of each layer
            and the dropout mask used on each layer.
    """
    cache = {"A0": X}

    for layer_index in range(L):
        a_previous = cache["A" + str(layer_index)]
        w = weights["W" + str(layer_index + 1)]
        b = weights["b" + str(layer_index + 1)]
        Z = np.dot(w, a_previous) + b
        dropout_mask = np.random.binomial(1, keep_prob, size=Z.shape)

        if layer_index == L - 1:
            # Softmax Activation Function for Output Layer
            softmax = np.exp(Z)
            cache["A" + str(layer_index + 1)] = softmax / \
                np.sum(softmax, axis=0, keepdims=True)
        else:
            # Sigmoid Activation Function for Hidden layers
            a_next = np.tanh(Z) * dropout_mask
            cache["A" + str(layer_index + 1)] = a_next / keep_prob
            cache["D" + str(layer_index + 1)] = dropout_mask

    return cache
