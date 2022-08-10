#!/usr/bin/env python3
"""
    Script that determines if a markov chain is absorbing
"""

import numpy as np


def check_column(P, current):
    """
        Check if the current state is accessible from the absorbing state

        Args:
            P (np.array): matrix of probabilities
            current (int): index of the current state

        Returns:
            bool: True if the current state is accessible
                       from the absorbing state,
                  False otherwise
    """
    column_values = P.T[current]

    return True if len(column_values[column_values > 0]) > 0 else False


def add_column_index(P, absorbing_index, row):
    """
        Add the index of the current state to the list of absorbing states

        Args:
            P (np.array): matrix of probabilities
            absorbing_index (list): list of absorbing states
            row (int): index of the current state

        Returns:
            list: list of absorbing states
    """
    n, _ = P.shape
    column = P.T[row]

    for i in range(n):
        if column[i] > 0 and i not in absorbing_index:
            absorbing_index = np.append(absorbing_index, i)

    return absorbing_index


def absorbing(P):
    """
        Check if the matrix of probabilities is absorbing

        Args:
            P (np.array): matrix of probabilities

        Returns:
            bool: True if the matrix of probabilities is absorbing,
                  False otherwise
    """
    try:
        if not isinstance(P, np.ndarray):
            raise TypeError("P must be a numpy.ndarray")

        if P.ndim != 2:
            raise TypeError("P must be a 2D matrix")

        if P.shape[0] != P.shape[1]:
            raise ValueError("P must be a square matrix")

        if np.any(np.sum(P, axis=1) != 1):
            raise ValueError("The rows of P must sum to 1")

        n, _ = P.shape
        diag = np.diag(P)

        # Case 1: each state is an absorbing state in the markov
        # (all terme in diagonal are 1)
        if np.all(diag == 1):
            return True

        # Case 2: each state is an absorbing state in the markov
        # (all terme in diagonal are 0)
        if not np.any(diag == 1):
            return False

        absorbing_state_index = np.where(diag == 1)[0]

        for row in range(n):
            # Case 3: each state is an absorbing state in the markov
            # The absorbing state is accessible from the current state
            # (at least one terme in column is greater than 0)
            if row in absorbing_state_index and check_column(P, row):
                absorbing_state_index = add_column_index(
                    P, absorbing_state_index, row)

        return len(absorbing_state_index) == n
    except Exception as e:
        return None
