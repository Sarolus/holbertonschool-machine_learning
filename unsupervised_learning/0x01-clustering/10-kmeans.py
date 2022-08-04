#!/usr/bin/env python3
"""
    Script that performs the K-means on a dataset.
"""

import sklearn.cluster as skc


def kmeans(X, k):
    """
        Performs the K-means on a dataset.

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            k: number of clusters

        Returns:
            centroids: numpy.ndarray of shape (k, d) containing the centroids
            idx: numpy.ndarray of shape (n,) containing the index of the
                cluster in each sample
    """

    kmeans = skc.KMeans(n_clusters=k)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    idx = kmeans.labels_

    return centroids, idx
