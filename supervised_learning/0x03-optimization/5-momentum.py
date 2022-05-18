#!/usr/bin/env python3
"""
    Momentum Optimizer Module
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
        Performs the update with momentum optimization

        Args:
            alpha: the learning rate
            beta1: the momentum weight
            var: the variable to be updated
            grad: the gradient at the current step
            v: the previous first moment of var

        Returns:
            the updated variable and the new v
    """

    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v

    return var, v
