#!/usr/bin/env python3

from machine_learning.train_model import BaseTrainModel
from financial.dataset import CSVDataSet


class ValueTrainModel(BaseTrainModel):
    def __init__(self, model_path='time_series_model.h5', csv_file_path='bitstamp.csv', dataset_type='csv'):
        if dataset_type == 'csv':
            self.dataset = CSVDataSet(csv_file_path=csv_file_path)
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        super().__init__(model_path=model_path)

    def dataset_graph(self):
        self.dataset.spliting_graph()
