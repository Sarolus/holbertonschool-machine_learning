#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

dataframe = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


def convert_to_datetime(x):
    return x.date()


dataframe = dataframe.rename(columns={'Timestamp': 'Datetime'})
dataframe.Datetime = pd.to_datetime(dataframe.Datetime, unit='s')
dataframe = dataframe[["Datetime", "Close"]]


print(dataframe.tail())
