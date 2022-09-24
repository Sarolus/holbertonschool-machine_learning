#!/usr/bin/env python3
"""
    Script that calculates the scaled dot product attention
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
        Calculates the scaled dot product attention

        Args:
            Q: is a tensor with its last two dimensions as (..., seq_len_q, dk)
                containing the query matrix
            K: is a tensor with its last two dimensions as (..., seq_len_v, dk)
                containing the key matrix
            V: is a tensor with its last two dimensions as (..., seq_len_v, dv)
                containing the value matrix
            mask: is a tensor that can be broadcast into
                  (..., seq_len_q, seq_len_v) containing the optional mask, or
                  defaulted to None

        Returns: output, weights
            output: a tensor with its last two dimensions as
                    (..., seq_len_q, dv) containing the scaled dot product
                    attention
            weights: a tensor with its last two dimensions as
                     (..., seq_len_q, seq_len_v) containing the attention
                     weights
    """

    # Calculate the scaled dot product
    scaled_dot_product = tf.matmul(
        Q, K, transpose_b=True) / tf.math.sqrt(
            tf.cast(tf.shape(K)[-1], tf.float32)
    )

    # Apply the mask to the scaled dot product
    if mask is not None:
        scaled_dot_product += (mask * -1e9)

    # Calculate the weights
    weights = tf.nn.softmax(scaled_dot_product, axis=-1)

    # Calculate the output
    output = tf.matmul(weights, V)

    return output, weights
