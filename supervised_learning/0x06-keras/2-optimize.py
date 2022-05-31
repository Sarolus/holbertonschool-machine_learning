#!/usr/bin/env python3
"""
    Optimize the model using Adam
"""

import tensorflow.keras as keras


def optimize_model(network, alpha, beta1, beta2):
    """
        Optimize the model using Adam

        Args:
            network: model to optimize
            alpha: learning rate
            beta1: first Adam optimizer parameter
            beta2: second Adam optimizer parameter

        Returns:
            the optimized model
    """

    network.compile(
        optimizer=keras.optimizers.Adam(
            alpha,
            beta1,
            beta2
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
