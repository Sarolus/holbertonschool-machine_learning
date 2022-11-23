#!/usr/bin/env python3
"""
    From CSV file panda conversion module
"""

import pandas as pd


def from_file(filename, delimiter):
    """
        Convert a csv file to a dataframe.

        Args:
            filename (csv): CSV file that we wish
            to convert.
            delimiter (str): Delimeter used in CSV
            parsing.

        Returns:
            pd.DataFrame: The CSV file as pandas
            dataframe.
    """

    return pd.read_csv(filename, delimiter=delimiter)
