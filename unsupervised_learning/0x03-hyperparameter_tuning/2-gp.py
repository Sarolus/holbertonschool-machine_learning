#!/usr/bin/env python3
"""
    Gaussian process module.
"""

import numpy as np


class GaussianProcess:
    """
        Class represents a noiseless 1D Gaussian process

        Attributes:
            X (np.ndarray): the inputs already sampled with the black-box
                            function
            Y (np.ndarray): the outputs of the black-box function for each
                            input in X
            l (float): the length parameter for the kernel
            sigma_f (float): the standard deviation given to the output of
                             the black-box function
            K (np.ndarray): the covariance kernel matrix
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
            Initializes the gaussian process.

            Args:
                X_init (np.ndarray): Array of shape (t, 1) with the t training
                                     data points
                Y_init (np.ndarray): Array of shape (t, 1) with the t training
                                     data labels
                l (float): Length-scale
                sigma_f (float): Noise variance
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
            Calculates the covariance kernel matrix between two matrices

            Args:
                X1 (np.ndarray): Array of shape (n, 1) with the first set of
                                 inputs
                X2 (np.ndarray): Array of shape (m, 1) with the second set of
                                 inputs

            Returns:
                (np.ndarray): Array of shape (n, m) with the covariance kernel
                              matrix
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

        return self.sigma_f**2 * np.exp(-.5 * (1 / self.l**2) * sqdist)

    def predict(self, X_s):
        """
            Predicts the mean and standard deviation of the Gaussian process at
            the points X_s.

            Args:
                X_s (np.ndarray): Array of shape (n, 1) containing the points
                                  at which the mean and standard deviation
                                  should be calculated

            Returns:
                (tuple of np.ndarray): Mean and standard deviation of the
                                       Gaussian process at the points X_s
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = np.dot(np.dot(K_s.T, K_inv), self.Y).reshape((X_s.shape[0]))
        cov_s = K_ss - K_s.T @ K_inv @ K_s
        std_s = np.diagonal(cov_s)

        return mu_s, std_s

    def update(self, X_new, Y_new):
        """
            Updates the Gaussian Process

            Args:
                X_new (np.ndarray): Array of shape (t', 1) with the new inputs
                Y_new (np.ndarray): Array of shape (t', 1) with the new outputs
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
