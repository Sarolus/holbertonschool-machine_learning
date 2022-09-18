#!/usr/bin/env python3

from django.core.management.base import BaseCommand
from financial.train_model import ValueTrainModel
import logging
import matplotlib.pyplot as plt
import pandas as pd


class Command(BaseCommand):
    logger = logging.getLogger(__name__)
    help = 'Trains the machine learning models'

    def add_arguments(self, parser):
        parser.add_argument('--epochs', '-e', type=int, help='Number of epochs', default=10)
        parser.add_argument('--dataset-type', '-dst', type=str, help='Dataset type', default='csv')
        parser.add_argument('--model-type', '-m', type=str, help='Model type', default='simple')
        parser.add_argument('--model_path', '-mp', type=str, help='Model path', default='time_series_model.h5')
        parser.add_argument('--csv_file_path', '-cfp', type=str, help='CSV file path', default='bitstamp.csv')
        parser.add_argument('--graphs', '-g', type=str, action='append', help='Graphs', default=[])

    def handle(self, *args, **options):
        self.logger.info('Training the machine learning models\n')

        epochs = options['epochs']
        dataset_type = options['dataset_type']
        model_type = options['model_type']
        model_path = options['model_path']
        csv_file_path = options['csv_file_path']
        graphs = options['graphs']

        train_model = ValueTrainModel(model_path=model_path, csv_file_path=csv_file_path, dataset_type=dataset_type)

        (train_ds, _), model, history = train_model.run(epochs=epochs, model_type=model_type)

        if 'dataset' in graphs:
            train_model.dataset_graph()

        if 'performance' in graphs:
            train_model.performance_graph(history)

        if 'resume' in graphs:
            train_model.resumme_graph(model, train_ds)
