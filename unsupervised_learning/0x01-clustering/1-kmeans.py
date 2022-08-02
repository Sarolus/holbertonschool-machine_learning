#!/usr/bin/env python3
"""
    Script that performs K-means on the dataset
"""

import numpy as np


def initialize(X, k):
    """
        Initializes cluster centroids for K-means

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            k: int representing the number of clusters
        Returns:
            centroids: numpy.ndarray of shape (k, d) containing the centroids
    """

    try:
        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            return ValueError("k must be greater than 0")

        _, dimensions = X.shape
        min_value = np.ndarray.min(X, axis=0)
        max_value = np.ndarray.max(X, axis=0)

        # Draw samples from a uniform distribution.
        # Samples are uniformly distributed over the half-open interval
        # [low, high) (includes low, but excludes high)
        centroids = np.random.uniform(
            low=min_value, high=max_value, size=(k, dimensions)
        )

        return centroids
    except Exception as e:
        return None


def get_distances(X, centroids):
    """
        Calculates the distance between each point in X and the centroids

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            centroids: numpy.ndarray of shape (k, d) containing the centroids

        Returns:
            distances: numpy.ndarray of shape (n, k) containing the distance
    """

    extended_centroids = centroids[:, np.newaxis]
    distances = np.sqrt(
        np.sum(np.square(X - extended_centroids), axis=2))

    return distances


def update_centroids(X, labels, centroids, distance_index):
    """
        Updates the centroids of the clusters

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            labels: numpy.ndarray of shape (n,) containing the index
                    of the cluster in centroids
            centroids: numpy.ndarray of shape (k, d) containing the centroids
                       of the clusters
            distance_index: int representing the index of the cluster in
            centroids

        Returns:
            centroids: numpy.ndarray of shape (k, d) containing the centroids
    """

    if X[labels == distance_index].size == 0:
        centroids[distance_index] = initialize(X, 1)
    else:
        centroids[distance_index] = np.mean(
            X[labels == distance_index], axis=0)

    return centroids


def kmeans(X, k, iterations=1000):
    """
        Performs K-means on a dataset

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            k: int representing the number of clusters
            iterations: int representing the number of iterations to perform

        Returns:
            centroids: numpy.ndarray of shape (k, d) containing the centroids
                       of the clusters
            labels: numpy.ndarray of shape (n,) containing the index
                    of the cluster in C
    """
    try:
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be greater than 0")

        centroids = initialize(X, k)

        if (centroids is None):
            raise ValueError("C is not initialized")

        old_centroids = centroids.copy()

        for _ in range(iterations):
            distances = get_distances(X, centroids)
            labels = np.argmin(distances, axis=0)

            for distance_index in range(k):
                centroids = update_centroids(
                    X, labels, centroids, distance_index)

            # If no change in the cluster centroids occurs between iterations,
            # your function should return
            if np.all(centroids == old_centroids):
                raise ValueError("Centroids not changed")

        return centroids, labels
    except Exception as e:
        return None, None
