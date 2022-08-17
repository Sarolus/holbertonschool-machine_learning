#!/usr/bin/env python3
"""
    Bayesian optimization module.
"""

import numpy as np
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
