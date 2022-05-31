#!/usr/bin/env python3
"""
    Script to save and load a keras model
"""

import tensorflow.keras as keras


def save_model(network, filename):
    """
        Saves a model

        Args:
            network: model to save
            filename: name of the file to save
    """

    network.save(filename)


def load_model(filename):
    """
        Loads a model

        Args:
            filename: name of the file to load

        Returns:
            the loaded model
    """

    return keras.models.load_model(filename)
