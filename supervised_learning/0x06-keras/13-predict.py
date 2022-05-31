#!/usr/bin/env python3


def predict(network, data, verbose=False):
    """
        Predicts the labels of a data set

        Args:
            network: model to predict with
            data: data to predict the labels of
            verbose: print information about the prediction or not

        Returns:
            the predicted labels
    """

    return network.predict(
        data,
        verbose=verbose,
    )
