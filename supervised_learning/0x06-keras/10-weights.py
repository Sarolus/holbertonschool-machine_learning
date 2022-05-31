#!/usr/bin/env python3
"""
    Script to save and load weights a keras model
"""


def save_weights(network, filename, save_format='h5'):
    """
        Saves the weights of a model

        Args:
            network: model whose weights will be saved
            filename: name of the file to save
            save_format: format to save the weights
    """

    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
        Loads the weights of a model

        Args:
            network: model whose weights will be loaded
            filename: name of the file to load
    """

    network.load_weights(filename)
