#!/usr/bin/env python3
"""
    Tensorflow Forward Propagation Module
"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
        Creates the forward propagation graph for the neural
        network.

        Args:
            x : _description_
            layer_sizes (list): List containing the number of nodes
            in each layer of the network. Defaults to [].
            activations (list): List containing the activation functions
            for each layer of the network. Defaults to [].

        Returns:
            Returns the prediction of the network in tensor form.
    """
    y_pred = x

    for index in range(len(layer_sizes)):
        y_pred = create_layer(x, layer_sizes[index], activations[index])

    return y_pred
