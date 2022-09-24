#!/usr/bin/env python3
"""
    Script that creates the decoder for a transformer
"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
        Creates the decoder for a transformer

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
        self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1
    ):
        """
            Constructor

            Args:
                N: is the number of blocks in the encoder
                dm: is the dimensionality of the model
                h: is the number of heads
                hidden: is the number of hidden units in the fully connected
                        layer
                target_vocab: is the size of the input vocabulary
                max_seq_len: is the maximum sequence length possible
                drop_rate: the dropout rate
        """

        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
            Public instance method

            Args:
                x: is a tensor of shape (batch, target_seq_len, dm) containing
                   the input to the decoder
                encoder_output: is a tensor of shape (batch, input_seq_len, dm)
                                containing the output of the encoder
                training: a boolean to determine if the model is training
                look_ahead_mask: the mask to be applied to the first multi head
                                 attention layer
                padding_mask: the mask to be applied to the second multi head
                              attention layer

            Returns:
                a tensor of shape (batch, target_seq_len, dm) containing the
                decoder output
        """

        seq_len = x.shape[1]
        embedding = self.embedding(x)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedding += self.positional_encoding[:seq_len]
        output = self.dropout(embedding, training=training)

        for i in range(self.N):
            output = self.blocks[i](output, encoder_output, training,
                                    look_ahead_mask, padding_mask)

        return output
