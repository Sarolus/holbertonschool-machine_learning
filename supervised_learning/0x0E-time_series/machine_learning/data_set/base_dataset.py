#!/usr/bin/env python3

from asyncio.log import logger
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

TRAIN_SPLITING_RATE = 0.8
TEST_SPLITING_RATE = 0.2
VALID_SPLITING_RATE = 0.1


class BaseDataSet:
    dataset = None
    logger = logging.getLogger(__name__)

    def cleaning(self):
        """
            Cleans the data set
        """

        raise NotImplementedError

    def curating(self):
        """
            Curates the data set
        """

        raise NotImplementedError

    def build(self):
        """
            Builds the data set
        """

        raise NotImplementedError

    def spliting(self):
        """
            Preprocesses the data set

            Returns:
                train_set: The train set
                test_set: The test set
        """

        try:
            self.logger.info('Spliting the data set\n')

            size = len(self.dataset)
            train_limit = int(size * TRAIN_SPLITING_RATE)

            train = self.dataset[:train_limit]
            test = self.dataset[train_limit:]

            self.logger.debug(
                f"Train data: {round(len(train) / size * 100, 2)}%")
            self.logger.debug(
                f"Test data: {round(len(test) / size * 100, 2)}%\n")

            return train, test
        except Exception as e:
            self.logger.error(f"Error: {e}")

    def _preprocessing(self):
        """
            Preprocesses the data set
        """

        try:
            self.cleaning()
            self.curating()
            self.dataset = self.dataframe.copy()
            self.train_data, self.test_data = self.spliting()

            train = BaseDataSet._normalize(self.train_data, 'train')
            test = BaseDataSet._normalize(self.test_data, 'test')

            return train, test
        except Exception as e:
            self.logger.error(f'Preprocessing error: {e}')

    def run(self):
        """
            Runs the data set
        """

        try:
            train, test = self._preprocessing()
            train = self._format_dataset(train)
            test = self._format_dataset(test)

            return train, test
        except Exception as e:
            self.logger.error(f'Run error: {e}')

    def _build_simple_model(self):
        """
            Builds a simple model
            https://www.tensorflow.org/tutorials/structured_data/time_series#5_train_the_model
            https://www.tensorflow.org/tutorials/structured_data/time_series#recurrent_neural_network
        """

        return tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(1)
        ])

    @classmethod
    def _normalization_constants(cls, data: np.ndarray):
        try:
            mean = np.mean(data, axis=0)
            standard_deviation = np.std(data, axis=0)

            return mean, standard_deviation
        except Exception as e:
            cls.logger.error(f'Normalization constant error: {e}')

    @classmethod
    def _normalize(cls, data: np.ndarray, data_name: str = ''):
        try:
            data_name = data_name + ' ' if data_name != '' else data_name
            cls.logger.info(f'Normalizing the {data_name}data set\n')
            mean, standard_deviation = cls._normalization_constants(data)
            data = (data - mean) / standard_deviation

            return data
        except Exception as e:
            cls.logger.error(f'Normalization error: {e}')

    def _format_dataset(self, data):
        """
            Formats the data set
            https://www.tensorflow.org/tutorials/structured_data/time_series#4_create_tfdatadatasets

            Args:
                data: The data set

            Returns:
                The formatted data set
        """

        def split_window(data):
            """
                Splits the data set into windows
                https://www.tensorflow.org/tutorials/structured_data/time_series
                https://www.tensorflow.org/tutorials/structured_data/time_series#2_split
            """

            try:
                input_slice = slice(0, 24)
                output_slice = slice(24, None)
                inputs = data[:, input_slice, :]
                outputs = data[:, output_slice, :]

                return inputs, outputs
            except Exception as e:
                self.logger.error(f'Split window error: {e}')

        self.logger.info("TEST")
        array = np.array(data, dtype=np.float32)
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=array,
            targets=None,
            sequence_length=24 + 1,  # 24 hours before + 1 hour to predict
            sequence_stride=1,
            shuffle=True,
            batch_size=256,
        )
        dataset = dataset.map(split_window)
        self.logger.info("TEST")

        return dataset

    def spliting_graph(self):
        raise NotImplementedError
