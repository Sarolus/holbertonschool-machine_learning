#!/usr/bin/env python3
"""
    Adam Optimizer Module
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
        Updates a variable in place using the Adam Optimization Algorithm

        Args:
            alpha: The learning rate.
            beta1: The weight used for the first moment.
            beta2: The weight used for the second moment.
            epsilon: Small number to avoid div by zero.
            var: np.ndarray containing the variable to be updated.
            grad: np.ndarray containing the gradient of var.
            v: The previous first moment of var.
            s: The previous second moment of var.
            t: The time step used for bias correction.

        Returns:
            The updated variable, the new first moment, and the new
            second moment, respectively.
    """

    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * np.power(grad, 2)
    v_hat = v / (1 - np.power(beta1, t))
    s_hat = s / (1 - np.power(beta2, t))
    var = var - alpha * v_hat / (np.sqrt(s_hat) + epsilon)

    return var, v, s
