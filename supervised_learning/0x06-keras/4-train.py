#!/usr/bin/env python3
"""
    Trains a model using mini-batch gradient descent
"""
import tensorflow.keras as keras


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                verbose=True,
                shuffle=False):
    """
        Trains a model using mini-batch gradient descent

        Args:
            network: model to train
            data: data to train the model
            labels: labels of data
            batch_size: size of the batch
            epochs: number of epochs to train the model
            verbose: print information about the training or not
            shuffle: shuffle the data before training or not

        Returns:
            the trained model
    """

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
    )
