#!/usr/bin/env python3
"""
    Batch Normalization Module
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
        Creates the operation to perform the batch normalization

        Args:
            Z: numpy.ndarray of shape (m, n) that contains the output of
                the previous operation
            gamma: numpy.ndarray of shape (1, n) containing the scales
            beta: numpy.ndarray of shape (1, n) containing the offsets
            epsilon: small number used to avoid division by zero

        Returns:
            Z_BN: the Batch Normalization of Z
    """

    Z_BN = (Z - np.mean(Z, axis=0)) / (np.sqrt(np.var(Z, axis=0) + epsilon))
    Z_BN = gamma * Z_BN + beta

    return Z_BN
