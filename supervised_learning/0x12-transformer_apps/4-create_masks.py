#!/usr/bin/env python3
"""
    Script that creates all masks for training/validation
"""

import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
        Creates all masks for training/validation

        Args:
            inputs: tf.Tensor of shape (batch_size, seq_len_in) that contains
                    the input sentence
            target: tf.Tensor of shape (batch_size, seq_len_out) that contains
                    the target sentence

        Returns:
            encoder_mask (tf.Tensor): padding mask of shape (batch_size, 1, 1,
                                      seq_len_in) to be applied in the encoder
            combined_mask (tf.Tensor): look ahead mask of shape (batch_size, 1,
                                       seq_len_out, seq_len_out) to be applied
                                       in the decoder
            decoder_mask (tf.Tensor): padding mask of shape (batch_size, 1, 1,
                                      seq_len_in) used in the 2nd attention
                                      block in the decoder
    """

    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the
    # decoder.
    # 1. get seq_len_out of target
    seq_len_out = target.shape[1]
    # 2. create the look ahead mask
    look_ahead_mask = 1 - \
        tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)
    # 3. create the decoder target padding mask
    decoder_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_target_padding_mask = decoder_target_padding_mask[:,
                                                              tf.newaxis, tf.newaxis, :]
    # 4. add the decoder target padding mask to the look ahead mask
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
