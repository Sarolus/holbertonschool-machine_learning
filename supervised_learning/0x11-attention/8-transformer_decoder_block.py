#!/usr/bin/env python3
"""
    Script that creates the decoder block for a transformer
"""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
        Create the decoder block for a transformer

        Attributes:
            mha1 (MultiHeadAttention): the first multi head attention layer
            mha2 (MultiHeadAttention): the second multi head attention layer
            dense_hidden (Dense): the first dense layer with hidden units and
                                  relu activation
            dense_output (Dense): the second dense layer with units units
            layernorm1 (LayerNormalization): the first layer normalization
                                             layer, with epsilon=1e-6
            layernorm2 (LayerNormalization): the second layer normalization
                                             layer, with epsilon=1e-6
            layernorm3 (LayerNormalization): the third layer normalization
                                             layer, with epsilon=1e-6
            dropout1 (Dropout): the first dropout layer
            dropout2 (Dropout): the second dropout layer
            dropout3 (Dropout): the third dropout layer
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

        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask=None,
             padding_mask=None):
        """
            Public instance method

            Args:
                x: is a tensor of shape (batch, target_seq_len, dm)containing
                   the input to the decoder block
                encoder_output: is a tensor of shape (batch,
        """

        # Masked multi-head attention (look ahead)
        mha1, _ = self.mha1(x, x, x, look_ahead_mask)
        mha1 = self.dropout1(mha1, training=training)
        out1 = self.layernorm1(mha1 + x)

        # Masked multi-head attention (padding)
        mha2, _ = self.mha2(out1, encoder_output, encoder_output,
                            padding_mask)
        mha2 = self.dropout2(mha2, training=training)
        out2 = self.layernorm2(mha2 + out1)

        # Feed forward network
        hidden = self.dense_hidden(out2)
        output = self.dense_output(hidden)
        output = self.dropout3(output, training=training)
        out3 = self.layernorm3(output + out2)

        return out3
