#!/usr/bin/env python3
"""
    Script that decode for machine translation
"""

import tensorflow as tf


class RNNDecoder(tf.keras.layers.Layer):
    """
        Decode for machine translation

        Attributes:
            embedding (Embedding): the embedding layer to be used for
                                   the decoder input
            gru (RNN): the RNN layer to be used for the decoder
            F (Dense): a Dense layer with vocab units, to be applied to
                       the decoder RNN output
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

        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab,
                                                   embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
            Public instance method

            Args:
                x: is a tensor of shape (batch, 1) containing the previous
                    word in the target sequence as an index of the target
                    vocabulary
                s_prev: is a tensor of shape (batch, units) containing the
                        previous decoder hidden state
                hidden_states: is a tensor of shape (batch, input_seq_len,
                               units)containing the outputs of the encoder

            Returns:
                y (tensor): a tensor of shape (batch, vocab) containing the
                            output word as a one hot vector in the target
                            vocabulary
                s (tensor): a tensor of shape (batch, units) containing the
                            new decoder hidden state
        """

        embed = self.embedding(x)
        output, s = self.gru(embed, initial_state=s_prev)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)

        return y, s
