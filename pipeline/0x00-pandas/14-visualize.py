#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from_file = __import__('2-from_file').from_file


dataframe = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

plt.close('all')
dataframe = dataframe.drop('Weighted_Price', axis=1)
dataframe = dataframe[::1440]
dataframe.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
dataframe['Datetime'] = pd.to_datetime(dataframe['Datetime'], unit='s')
dataframe = dataframe[dataframe['Datetime'] > '2017']
dataframe.fillna({'Volume_(BTC)': 0, 'Volume_(Currency)': 0}, inplace=True)
dataframe['Close'].ffill(inplace=True)
dataframe.fillna({'Open': dataframe['Close'], 'High': dataframe['Close'], 'Low': dataframe['Close']},
                 inplace=True)
dataframe.set_index('Datetime', inplace=True)

dataframe.plot()
plt.savefig("14-graph")
