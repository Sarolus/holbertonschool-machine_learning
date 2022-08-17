#!/usr/bin/env python3
"""
    Bayesian optimization module.
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class that performs Bayesian optimization on a noiseless 1D Gaussian
    process.

    Attributes:
        f (function): Function to be optimized
        gp (GaussianProcess): Gaussian process object
        X_s (np.ndarray):  all acquisition sample points, evenly spaced
                            between min and max
        xsi (float): Exploration/exploitation factor for the acquisition
                     function
        minimize (bool): True if the objective function is to be minimized,
                            False if it is to be maximized
    """

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True
    ):
        """
        Initializes the Bayesian optimization object.

        Args:
            f (function): Function to be optimized
            X_init (np.ndarray): Array of shape (t, 1) with the t training
                                    data points
            Y_init (np.ndarray): Array of shape (t, 1) with the t training
                                    data labels
            bounds (tuple): Bounds for the parameters of the function
            ac_samples (int): Number of samples for the acquisition function
            l (float): Length-scale
            sigma_f (float): Noise variance
            xsi (float): Exploration/exploitation factor for the acquisition
                            function
            minimize (bool): True if the objective function is to be minimized,
                                False if it is to be maximized
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location.

        Returns:
            x_next (np.ndarray): Array of shape (1, 1) with the next best point
            EI (np.ndarray): expected improvement of each potential sample
        """

        # Retrieve mean and standard deviation of the acquisition sample points
        mean, std_deviation = self.gp.predict(self.X_s)

        if self.minimize:
            # Calculates the reduced mean
            scaled_mean = np.min(self.gp.Y) - mean

        else:
            # Calculates the increased mean
            scaled_mean = mean - np.max(self.gp.Y)

        # Calculates the improvement variable
        improvement = scaled_mean - self.xsi

        Z = np.where(std_deviation == 0, 0, improvement / std_deviation)

        # Expected Improvement Acquisition Formula
        expected_improvement = improvement * \
            norm.cdf(Z) + std_deviation * norm.pdf(Z)

        x_next = self.X_s[np.argmax(expected_improvement)]

        return x_next, expected_improvement
