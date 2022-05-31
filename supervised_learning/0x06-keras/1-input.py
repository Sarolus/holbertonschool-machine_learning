#!/usr/bin/env python3
"""
    Builds a neural network with the Keras library.
"""

import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        Builds a neural network with the Keras library.

        Args:
            nx: is the number of input features to the neuron.
            layers: is a list representing the number of nodes in each layer.
            activations: is a list representing the activation functions for
            each layer.
            lambtha: is the L2 regularization parameter.
            keep_prob: is the probability that a node will be kept.

        Returns:
            A Keras model instance.
    """

    # Create the input
    input = keras.Input(shape=(nx,))

    for i in range(len(layers)):
        # Add the layer
        if i == 0:
            # first layer
            layer = keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=keras.regularizers.l2(lambtha)
            )(input)
        else:
            # other layers
            layer = keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=keras.regularizers.l2(lambtha)
            )(layer)

        # Add the dropout
        if i < len(layers) - 1:
            layer = keras.layers.Dropout(1 - keep_prob)(layer)

    # Return the model
    return keras.Model(inputs=input, outputs=layer)
