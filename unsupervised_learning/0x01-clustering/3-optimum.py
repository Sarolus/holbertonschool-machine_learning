#!/usr/bin/env python3
"""
    K-mean Optimum Number of Cluster Calculation
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
        Tests for the optimum number of clusters by variance.

        Args:
            X (np.ndarray): The data set.
            kmin (int, optional): Minimum number of clusters to check for.
            Defaults to 1.
            kmax (_type_, optional): Maximum number of clusters to check for.
            Defaults to 1.
            iterations (int, optional): Maximum number of iterations for
            K-means. Defaults to 1000.

        Returns:
            lists: The k_means of each cluster size and the difference variance
            from the smallest cluster size for each cluster size.
    """
    try:

        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(kmin, int) or kmin < 1:
            return None, None

        if (kmax is None):
            kmax = X.shape[0]

        if not isinstance(kmax, int) or kmax < 1:
            return None, None

        if not isinstance(iterations, int) or iterations < 1:
            return None, None

        if kmax - kmin < 2:
            return None, None

        k_means = []
        variance_differences = []

        for k in range(kmin, kmax + 1):

            centroids, labels = kmeans(X, k)

            if k == kmin:
                kmin_variance = variance(X, centroids)

            k_variance = variance(X, centroids)
            var_diff = kmin_variance - k_variance

            variance_differences.append(var_diff)
            k_means.append((centroids, labels))

        return k_means, variance_differences
    except Exception as exception:
        return None, None
