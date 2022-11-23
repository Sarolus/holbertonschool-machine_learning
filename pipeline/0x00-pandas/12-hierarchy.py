#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file


first_dataframe = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
second_dataframe = from_file(
    'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

first_dataframe.set_index('Timestamp', inplace=True)
second_dataframe.set_index('Timestamp', inplace=True)
first_dataframe = first_dataframe.loc[1417411980:1417417980]
second_dataframe = second_dataframe.loc[1417411980:1417417980]
dataframe = pd.concat([second_dataframe, first_dataframe],
                      keys=['bitstamp', 'coinbase'])
dataframe = dataframe.swaplevel(0, 1)
dataframe.sort_index(inplace=True)

print(dataframe)
