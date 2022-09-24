#!/usr/bin/env python3
"""
    Script that creates the encoder for a transformer
"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
        Creates the encoder for a transformer

        Attributes:
            N (int): the number of blocks in the encoder
            dm (int): the dimensionality of the model
            embedding (Dense): the embedding layer for the inputs
            positional_encoding (numpy.ndarray): a numpy.ndarray of shape
                                                (max_seq_len, dm) containing
                                                the positional encodings
            blocks (list): a list of length N containing all of the
                           EncoderBlock‘s
            dropout (Dropout): the dropout layer, to be applied to the
                               positional encodings
    """

    def __init__(
        self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1
    ):
        """
            Constructor

            Args:
                N: is the number of blocks in the encoder
                dm: is the dimensionality of the model
                h: is the number of heads
                hidden: is the number of hidden units in the fully connected
                        layer
                input_vocab: is the size of the input vocabulary
                max_seq_len: is the maximum sequence length possible
                drop_rate: the dropout rate
        """

        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
            Public instance method

            Args:
                x: is a tensor of shape (batch, input_seq_len, dm) containing
                   the input to the encoder
                training: a boolean to determine if the model is training
                mask: the mask to be applied for multi head attention
        """

        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)
        return x
