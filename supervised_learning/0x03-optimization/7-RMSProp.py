#!/usr/bin/env python3
"""
    RMS Prop Optimizer Module
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
        Performs the update with RMSProp optimization

        Args:
            alpha: the learning rate
            beta2: the RMSProp weight
            epsilon: small number to avoid division by zero
            var: the variable to be updated
            grad: the gradient at the current step
            s: the previous second moment of var

        Returns:
            the updated variable and the new s
    """

    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
