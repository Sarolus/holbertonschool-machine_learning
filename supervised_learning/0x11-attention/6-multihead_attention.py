#!/usr/bin/env python3
"""
    Script that perform multi head attention
"""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
        Perform multi head attention

        Attributes:
            h (int): the number of heads
            dm (int): the dimensionality of the model
            depth (int): the depth of each attention head
            Wq (Dense): the dense layer with dk units, to be applied to
                        the previous decoder hidden state
            Wk (Dense): the dense layer with dk units, to be applied to
                        the encoder hidden states
            Wv (Dense): the dense layer with dv units, to be applied to
                        the encoder hidden states
            linear (Dense): the final dense layer with dm units
    """

    def __init__(self, dm, h):
        """
            Constructor

            Args:
                dm: is an integer representing the dimensionality of the model
                h: is an integer representing the number of heads
        """

        super(MultiHeadAttention, self).__init__()

        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(units=self.dm)
        self.Wk = tf.keras.layers.Dense(units=self.dm)
        self.Wv = tf.keras.layers.Dense(units=self.dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """
            Public instance method

            Args:
                Q: is a tensor of shape (batch, seq_len_q, dk) containing
                    the input to generate the query matrix
                K: is a tensor of shape (batch, seq_len_v, dk) containing
                    the input to generate the key matrix
                V: is a tensor of shape (batch, seq_len_v, dv) containing
                    the input to generate the value matrix
                mask: is always None

            Returns:
                output: a tensor with its last two dimensions as
                        (..., seq_len_q, dm) containing the scaled dot product
                        attention
                weights: a tensor with its last three dimensions as
                         (..., h, seq_len_q, seq_len_v) containing the
                         attention  weights
        """

        # Calculate the number of batches
        batch = tf.shape(Q)[0]

        # Calculate the query, key and value
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # Split the query, key and value into self.h heads
        q = tf.reshape(q, (batch, -1, self.h, self.depth))
        k = tf.reshape(k, (batch, -1, self.h, self.depth))
        v = tf.reshape(v, (batch, -1, self.h, self.depth))

        # Transpose the query, key and value
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Calculate the scaled dot product attention
        scaled_attention, weights = sdp_attention(q, k, v, mask)

        # Concatenate the scaled_attention
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch, -1, self.dm))

        # Calculate the output
        output = self.linear(concat_attention)

        return output, weights
