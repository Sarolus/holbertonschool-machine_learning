#!/usr/bin/env python3
"""
    Policy Gradient Module
"""

import numpy as np


def policy(matrix, weights):
    """
        Computes to policy with a weight of a matrix

        Args:
            matrix (np.ndarray): The given matrix (states in our case).
            weights (np.ndarray): The given matrix of weights
    """
    
    Z_exp = np.exp(matrix.dot(weights))
    
    return Z_exp / np.sum(Z_exp)


def policy_gradient(state, weight):
    """
        Computes the monte-carlo policy gradient

    Args:
        state (np.ndarray): The given matrix of states.
        weight (np.ndarray): The given matrix of weights.
    """
    
    P = policy(state, weight)
    action = np.random.choice(len(P[0]), p=P[0])

    s = P.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)
    softmax = softmax[action, :]
    
    dlog = softmax / P[0, action]
    gradient = state.T.dot(dlog[None, :])
    
    return action, gradient
