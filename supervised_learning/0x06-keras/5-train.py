#!/usr/bin/env python3
"""
    Trains a model using mini-batch gradient descent
    to also analyse validation data
"""


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                verbose=True,
                shuffle=False):
    """
        Trains a model using mini-batch gradient descent
        to also analyse validation data

        Args:
            network: model to train
            data: data to train the model
            labels: labels of data
            batch_size: size of the batch
            epochs: number of epochs to train the model
            validation_data: data to validate the model with, if None,
            the model is not validated
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
        validation_data=validation_data,
    )
