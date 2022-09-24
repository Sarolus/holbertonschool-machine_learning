#!/usr/bin/env python3
"""
    Script that creates a transformer network
"""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """
        Creates a transformer network

        Attributes:
            encoder (Encoder): the encoder layer
            decoder (Decoder): the decoder layer
            linear (Dense): a final Dense layer with target_vocab units
    """

    def __init__(
        self, N, dm, h, hidden, input_vocab,
        target_vocab, max_seq_len, drop_rate=0.1
    ):
        """
            Constructor

            Args:
                N: is the number of blocks in the encoder and decoder
                dm: is the dimensionality of the model
                h: is the number of heads
                hidden: is the number of hidden units in the fully connected
                        layer
                input_vocab: is the size of the input vocabulary
                target_vocab: is the size of the target vocabulary
                max_seq_len: is the maximum sequence length possible
                drop_rate: the dropout rate
        """

        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_len,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_len,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
            Public instance method

            Args:
                inputs: is a tensor of shape (batch, input_seq_len, dm)
                        containing the inputs
                target: is a tensor of shape (batch, target_seq_len, dm)
                        containing the target
                training: a boolean to determine if the model is training
                encoder_mask: the padding mask to be applied to the encoder
                look_ahead_mask: the look ahead mask to be applied to the
                                 decoder
                decoder_mask: the padding mask to be applied to the decoder

            Returns:
                a tensor of shape (batch, target_seq_len, target_vocab)
                containing the transformer output
        """

        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        output = self.linear(decoder_output)

        return output
