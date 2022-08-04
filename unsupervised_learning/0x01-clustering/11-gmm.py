#!/usr/bin/env python3
"""
    Script that calculates a GMM from a dataset.
"""

import sklearn.mixture as skm


def gmm(X, k):
    """
        Calculates the Gaussian Mixture Model for a dataset.

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            k: number of clusters

        Returns:
            pi: numpy.ndarray of shape (k, 1) containing the probabilities
                for each cluster
            mean: numpy.ndarray of shape (k, d) containing the centroids
            covariance: numpy.ndarray of shape (k, d, d) containing
                        the covariances
            labels: numpy.ndarray of shape (n,) containing the index of the
                    cluster in each sample
            bic: float containing the Bayesian Information Criterion
    """

    gmm = skm.GaussianMixture(n_components=k)
    gmm.fit(X)
    pi = gmm.weights_
    mean = gmm.means_
    covariance = gmm.covariances_
    labels = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, mean, covariance, labels, bic
