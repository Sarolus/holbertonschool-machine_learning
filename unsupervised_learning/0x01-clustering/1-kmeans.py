#!/usr/bin/env python3
"""
    Script that perform K-means on a dataset.
"""

import numpy as np


def get_distances(centroids, X):
    """
        Get the distances between the centroids and the data.

        Args:
            centroids: numpy.ndarray of shape (k, d) containing the centroids
            X: numpy.ndarray of shape (n, d) containing the dataset

        Returns:
            distances: numpy.ndarray of shape (n, k) containing the distances
    """

    extended_centroids = centroids[:, np.newaxis]

    return np.sqrt(np.sum(np.square(extended_centroids - X), axis=2))


def update_centroids(
    X, labels, distance_index, centroids, low, high, dimensions
):
    """
        Update the centroids.

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            labels: numpy.ndarray of shape (n,) containing the labels of the
                clusters
            distance_index: int representing the index of the cluster
            centroids: numpy.ndarray of shape (k, d) containing the centroids
            low: numpy.ndarray of shape (d,) containing the lower bounds of
                 the dataset
            high: numpy.ndarray of shape (d,) containing the upper bounds of
                  the dataset
            dimensions: int representing the number of dimensions of the
                dataset

        Returns:
            centroids: numpy.ndarray of shape (k, d) containing the centroids
    """

    if (labels == distance_index).sum() == 0:
        centroids[distance_index] = np.random.uniform(
            low=low, high=high, size=(1, dimensions))
    else:
        centroids[distance_index] = np.mean(
            X[labels == distance_index], axis=0)

    return centroids


def kmeans(X, k, iterations=1000):
    """
    Method to perform K-means on a dataset.
    Parameters:
        X (numpy.ndarray of shape(n, d)): The dataset.
            n (int): number of data points.
            d (int): number of dimensions for each data point.
        K (positive int): The number of clusters.
        iterations (positive int): the maximum number of iterations
          that should be performed
    Returns:
        C, clss or None, None on failure
        C (numpy.ndarray of shape(k, d)):
        containing the centroid means for each cluster.
        clss (numpy.ndarray of shape (n,)):
        containing the index of the cluster in C
        that each data point belongs to
    """
    try:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")

        if X.ndim != 2:
            raise TypeError("X must be a 2D array")

        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be greater than 0")

        n, dimensions = X.shape
        low = np.amin(X, axis=0)
        high = np.amax(X, axis=0)
        centroids = np.random.uniform(low=low, high=high, size=(k, dimensions))

        for i in range(iterations):
            old_centroids = centroids.copy()
            labels = np.zeros(n)
            distances = get_distances(centroids, X)
            labels = np.argmin(distances, axis=0)

            for distance_index in range(k):
                centroids = update_centroids(
                    X, labels, distance_index, centroids, low, high, dimensions
                )

            distances = get_distances(centroids, X)
            labels = np.argmin(distances, axis=0)
            if np.array_equal(centroids, old_centroids):
                break
        return centroids, labels
    except Exception:
        return None, None
