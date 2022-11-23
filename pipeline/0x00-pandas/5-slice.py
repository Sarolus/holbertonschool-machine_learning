#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file


dataframe = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

dataframe = dataframe.drop(["Timestamp",
                            "Open",
                            "Low",
                            "Volume_(BTC)",
                            "Volume_(Currency)",
                            "Weighted_Price"],
                           axis=1).iloc[::60, :]

print(dataframe.tail())
