#!/usr/bin/env python3
"""
    Script that performs the agglomerative clustering on a dataset.
"""

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
        Performs the agglomerative clustering on a dataset.

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            dist: the maximum cophenetic distance for the clusters

        Returns:
            labels: numpy.ndarray of shape (n,) containing the index of the
                    cluster in each sample
    """

    Z = sch.linkage(X, 'ward')
    dn = sch.dendrogram(Z, color_threshold=dist)
    labels = sch.fcluster(Z, t=dist, criterion='distance')
    plt.show()
    return labels
