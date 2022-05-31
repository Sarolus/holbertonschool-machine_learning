#!/usr/bin/env python3
"""
    Script to save and load JSON configuration of a keras model
"""

import tensorflow.keras as keras


def save_config(network, filename):
    """
        Saves a model's configuration

        Args:
            network: model whose configuration will be saved
            filename: name of the file to save
    """

    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """
        Loads a model's configuration

        Args:
            filename: name of the file to load

        Returns:
            the loaded model
    """

    with open(filename, 'r') as f:
        return keras.models.model_from_json(f.read())
