#!/usr/bin/env python3
"""
    Gradient Descent Early Stopping Module
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
        Checks if the validation cost is the best so far and
        increments the counter

        Args:
            cost: validation cost at the current epoch
            opt_cost: lowest recorded validation cost
            threshold: minimum difference between the lowest recorded
                          validation cost and the current validation cost
                            to consider the model as improved
            patience: number of epochs to wait before early stopping
            count: number of epochs the model has been trained for

        Returns:
            a boolean value indicating if the model should be early stopped
    """

    if cost < opt_cost - threshold:
        count = 0
        opt_cost = cost
    else:
        count += 1

    return (True, count) if count >= patience else (False, count)
