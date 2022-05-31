#!/usr/bin/env python3
"""
    Script that tests a neural network.
"""
import tensorflow.keras as keras


def test_model(network, data, labels, verbose=True):
    """
        Tests a model using mini-batch gradient descent

        Args:
            network: model to test
            data: data to test the model
            labels: labels of data
            verbose: print information about the testing or not

        Returns:
            the tested model
    """

    return network.evaluate(
        data,
        labels,
        verbose=verbose,
    )
