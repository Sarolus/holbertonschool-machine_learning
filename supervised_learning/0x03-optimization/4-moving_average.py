#!/usr/bin/env python3
"""
    Moving Average Calculation Module
"""

import numpy as np


def moving_average(data, beta):
    """
        Calculates the moving average of data

        Args:
            data: 1D array of data
            beta: weight of the moving average

        Returns:
            1D array of the moving average of data
    """

    moving_average = []

    # Store cumulative sums of array in cum_sum array
    v = 0

    for index in range(len(data)):
        # Update cumulative sum
        v = beta * v + (1 - beta) * data[index]

        # Store the cumulative average of
        # current window in moving average list
        moving_average.append(v)

    exponential_moving_average = [
        moving_average[index] /
        (1 - beta ** (index + 1))
        for index in range(len(data))
    ]

    return exponential_moving_average
