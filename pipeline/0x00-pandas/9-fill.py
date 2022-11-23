#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file


dataframe = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

dataframe.drop('Weighted_Price', inplace=True, axis=1)
dataframe["Volume_(BTC)"].fillna(0, inplace=True)
dataframe["Volume_(Currency)"].fillna(0, inplace=True)
dataframe["High"] = dataframe["Close"].fillna(method="ffill")
dataframe["Low"] = dataframe["Close"].fillna(method="ffill")
dataframe["Open"] = dataframe["Close"].fillna(method="ffill")
dataframe["Close"] = dataframe["Close"].fillna(method="ffill")

print(dataframe.head())
print(dataframe.tail())
