#!/usr/bin/env python3
"""
    Script that creates an encoder block for a transformer
"""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
        Create an encoder block for a transformer

        Attributes:
            mha (MultiHeadAttention): the multi head attention layer
            dense_hidden (Dense): the first dense layer with hidden units
                                  and relu activation
            dense_output (Dense): the second dense layer with units units
            layernorm1 (LayerNormalization): the first layer normalization
                                             layer, with epsilon=1e-6
            layernorm2 (LayerNormalization): the second layer normalization
                                             layer, with epsilon=1e-6
            dropout1 (Dropout): the first dropout layer
            dropout2 (Dropout): the second dropout layer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
            Constructor

            Args:
                dm: is an integer representing the dimensionality of the model
                h: is an integer representing the number of heads
                hidden: is the number of hidden units in the fully connected
                        layer
                drop_rate: the dropout rate
        """

        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
            Public instance method

            Args:
                x: is a tensor of shape (batch, input_seq_len, dm)containing
                   the input to the encoder block
                training: a boolean to determine if the model is training
                mask: the mask to be applied for multi head attention

            Returns:
                a tensor of shape (batch, input_seq_len, dm) containing the
                block’s output
        """

        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)

        hidden = self.dense_hidden(out1)
        output = self.dense_output(hidden)
        output = self.dropout2(output, training=training)
        out2 = self.layernorm2(out1 + output)

        return out2
