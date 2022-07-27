#!/usr/bin/env python3
"""
    Shannon Entropy And P Affinities Calculation
"""
import numpy as np


def HP(Di, beta):
    """
        Calculates the Shannon entropy and P affinities
        relative to a data point.

        Args:
            Di (np.ndarray): Pairwise distance between a data
            point.
            beta (np.ndarray): Beta value for the gaussian
            distribution.

        Returns:
            The shannon entropy of the points and the np.ndarray
            that contains the P affinities of the points.
    """
    pairwise_distance = np.exp(-Di * beta)
    pairwise_distance /= np.sum(pairwise_distance)
    shannon_entropy = np.sum(-pairwise_distance * np.log2(pairwise_distance))

    return shannon_entropy, pairwise_distance
