#!/usr/bin/env python3

from machine_learning.data_set import BaseDataSet
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from os import path

class BaseTrainModel:
    logger = logging.getLogger(__name__)
    dataset:BaseDataSet = None

    def __init__(self, model_path='time_series_model.h5'):
        self.model_path = model_path

    def _build_model(self, model_type):
        """
            Builds the model

            Args:
                model_type: The model type

            Returns:
                model: The model
        """

        if path.exists(self.model_path):
            return tf.keras.models.load_model(self.model_path)
        elif model_type == 'simple':
            return self._build_simple_model()
        elif model_type == 'deep':
            return self._build_deep_model()
        elif model_type == 'ultra':
            return self._build_ultra_model()
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def _build_simple_model(self):
        """
            Builds a simple model
        """

        return tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(1)
        ])

    def _build_deep_model(self):
        """
            Builds a deep model
        """

        return tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dense(1)
        ])

    def _build_ultra_model(self):
        """
            Builds an ultra model
        """

        return tf.keras.models.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.3, recurrent_activation='relu'),
            tf.keras.layers.LSTM(256, return_sequences=False, recurrent_dropout=0.3),
            tf.keras.layers.Dense(1)
        ])

    def _build_callbacks(self):
        """
            Builds the callbacks
        """

        callbacks = []

        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        )

        return callbacks

    def _compile_and_fit(self, model, train_data, test_data, epochs=100):
        """
            Compiles and fits the model

            Args:
                model: The model
                train_data: The train data
                test_data: The test data

            Returns:
                history: The history
        """

        model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanSquaredError(), tf.metrics.Accuracy()]
        )

        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=test_data,
            callbacks=self._build_callbacks()
        )

        return history

    def run(self, epochs=100, model_type='simple'):
        """
            Runs the model training
        """

        try:
            train_data, test_data = self.dataset.run()
            model = self._build_model(model_type)
            history = self._compile_and_fit(model, train_data, test_data, epochs)
            model.save(filepath=self.model_path, save_format="h5")

            return (train_data, test_data), model, history
        except Exception as e:
            self.logger.error(f"Train model Error: {e}")

    def resumme_graph(self, model, train_ds):
        plt.figure("Resume", figsize=(10, 7))
        for b, (x, y) in enumerate(train_ds.take(4)):
            plt.subplot(2, 2, b+1)
            prediction = model.predict(x)[b]
            plt.plot(list(range(24)), x[b, :, -1]) # current data
            plt.plot(24, y[b].numpy(), 'r*', label="Label") # Label
            plt.plot(24, prediction, 'g^', label='Prediction') # prediction
            plt.title(f"Prediction #{b+1}")
        plt.tight_layout()
        plt.show()

    def performance_graph(self, history):
        plt.figure("Loss performance", figsize=(10, 7))
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.ylabel("mean squared error")
        plt.xlabel("epoch")
        plt.legend(['Train loss', 'Valid loss'], loc='upper left')
        plt.show()

    def spliting_graph(self):
        raise NotImplementedError
