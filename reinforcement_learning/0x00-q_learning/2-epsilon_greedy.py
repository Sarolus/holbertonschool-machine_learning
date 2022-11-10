#!/usr/bin/env python3
"""
    Script that uses the epsilon-greedy algorithm
    to determine the next action
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
        Uses epsilon-greedy to determine the next action

        Args:
            Q: is a numpy.ndarray containing the q-table
            state: is the current state
            epsilon: is the epsilon to use for the calculation

        Returns:
            the next action index
    """

    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])

    return np.argmax(Q[state])
