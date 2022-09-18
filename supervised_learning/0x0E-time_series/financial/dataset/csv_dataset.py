#!/usr/bin/env python3


from machine_learning.data_set.base_dataset import BaseDataSet, plt
import pandas as pd


class CSVDataSet(BaseDataSet):
    dataframe = None

    def cleaning(self):
        self.logger.info('Cleaning the data set\n')
        self.logger.info('Drop rows with NaN values')
        self.dataframe = self.dataframe.ffill()
        self.logger.info('Remove correlated feature columns')
        self.dataframe = self.dataframe.drop(columns=['Open', 'High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price'])

    def curating(self):
        self.logger.info('Curating the data set\n')
        self.logger.info('Format timestamp to datetime')
        self.dataframe["Timestamp"] = pd.to_datetime(self.dataframe["Timestamp"], unit='s')
        self.logger.info('Convert minutes to hours')
        self.dataframe = self.dataframe[8::60]
        self.logger.info('Define time index')
        self.dataframe = self.dataframe.set_index('Timestamp')

    def __init__(self, csv_file_path= 'bitstamp.csv'):
        self.logger.info('Loading the data set from CSV file\n')
        self.dataframe = pd.read_csv(csv_file_path)

    def spliting_graph(self):
        plt.figure("Spliting", figsize=(12, 8))
        plt.scatter(x=self.train_data.index, y=self.train_data["Close"], s=5, label="Train Data")
        plt.scatter(x=self.test_data.index, y=self.test_data["Close"], s=5, label="Test Data")
        plt.legend()
        plt.show()
