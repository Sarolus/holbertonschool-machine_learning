#!/usr/bin/env python3
"""
    Script that encode for machine translation
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
        Encode for machine translation

        Attributes:
            batch (int): the batch size
            units (int): the number of hidden units in the RNN cell
            embedding (Embedding): the embedding layer to be used for
                                   the encoder input
            gru (RNN): the RNN layer to be used for the encoder
    """

    def __init__(self, vocab, embedding, units, batch):
        """
            Constructor

            Args:
                vocab: is an instance of Vocabulary
                embedding: is an instance of Embedding
                units: is an integer representing the number of hidden units
                       in the RNN cell
                batch: is an integer representing the batch size
        """

        super(RNNEncoder, self).__init__()

        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab,
                                                   embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
            Initializes the hidden states for the RNN cell to a tensor of zeros

            Returns:
                a tensor of shape (batch, units)containing the initialized
                hidden states
        """

        initializer = tf.keras.initializers.Zeros()

        return initializer(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
            Public instance method

            Args:
                x: is a tensor of shape (batch, input_seq_len)containing the
                    input to the encoder layer as word indices within the
                    vocabulary
                initial: is a tensor of shape (batch, units) containing the
                    initial hidden state

            Returns:
                outputs, hidden
                    outputs: is a tensor of shape (batch, input_seq_len,
                        units)containing the outputs of the encoder
                    hidden: is a tensor of shape (batch, units) containing the
                        last hidden state of the encoder
        """

        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
