#!/usr/bin/env python3
"""
    From numpy panda conversion module
"""

import pandas as pd


def from_numpy(array):
    """
        Convert an np.ndarray to a dataframe.

        Args:
            array (np.ndarray): Numpy array that we wish
            to convert.

        Returns:
            pd.DataFrame: The converted numpy array as pandas
            dataframe.
    """

    return pd.DataFrame(array)
