#!/usr/bin/env python3
"""
    Script that calculate the attention for machine translation
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
        Calculate the attention for machine translation

        Attributes:
            W (Dense): a Dense layer with units, to be applied to
                       be applied to the previous decoder hidden state
            U (Dense): a Dense layer with units, to be applied to
                       be applied to the encoder hidden states
            V (Dense): a Dense layer with 1 units, to be applied to
                       be applied to the tanh of the sum of the outputs of
                       W and U
    """

    def __init__(self, units):
        """
            Constructor

            Args:
                units: is an integer representing the number of hidden units
                    in the alignment model
        """

        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
            Public instance method

            Args:
                s_prev: is a tensor of shape (batch, units) containing the
                    previous decoder hidden state
                hidden_states: is a tensor of shape (batch, input_seq_len,
                    units)containing the outputs of the encoder

            Returns:
                context (tensor): a tensor of shape (batch, units) that
                                  contains the context vector for the decoder
                weights (tensor): a tensor of shape (batch, input_seq_len, 1)
                                  that contains the attention weights
        """

        # Expand the shape of s_prev to be (batch, 1, units)
        s_prev = tf.expand_dims(s_prev, 1)

        # Calculate the score
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))

        # Calculate the weights
        weights = tf.nn.softmax(score, axis=1)

        # Calculate the context
        context = weights * hidden_states
        # Is used to find sum of elements across axes of a tensor
        context = tf.reduce_sum(context, axis=1)

        return context, weights
