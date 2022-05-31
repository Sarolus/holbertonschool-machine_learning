#!/usr/bin/env python3
"""
    Trains a model using mini-batch gradient descent
    to also train model using early stopping
"""

import tensorflow.keras as keras


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                verbose=True,
                shuffle=False):
    """
        Trains a model using mini-batch gradient descent
        to also train model using early stopping

        Args:
            network: model to train
            data: data to train the model
            labels: labels of data
            batch_size: size of the batch
            epochs: number of epochs to train the model
            validation_data: data to validate the model with, if None,
            the model is not validated
            early_stopping: use early stopping or not
            patience: patience for early stopping
            verbose: print information about the training or not
            shuffle: shuffle the data before training or not

        Returns:
            the trained model
    """

    callbacks = []
    if early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
        )
        callbacks.append(early_stopping)

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks,
    )
