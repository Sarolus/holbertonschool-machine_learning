#!/usr/bin/env python3
"""
    Inception Blocks Module
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
        Builds an inception block as described in Going Deeper
        with Convolutions (2014)

        Args:
            A_prev: tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            filters:
                F1: integer, number of filters in the 1x1 convolution
                F3R: integer, number of filters in the 3x3 convolution
                     in the 1x1 branch
                F3: integer, number of filters in the 3x3 convolution
                    in the 3x3 branch
                F5R: integer, number of filters in the 5x5 convolution
                     in the 1x1 branch
                F5: integer, number of filters in the 5x5 convolution
                    in the 3x3 branch
                FPP: integer, number of filters in the pooling layer

        Returns:
            A: output of the inception block
            parameters: list of parameters
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv_1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        activation="relu",
        padding='same',
    )(A_prev)

    # 3x3 convolution
    conv_3x3 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        activation="relu",
        padding='same',
    )(A_prev)

    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        activation="relu",
        padding='same',
    )(conv_3x3)

    # 5x5 convolution
    conv_5x5 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        activation="relu",
        padding='same',
    )(A_prev)

    conv_5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        activation="relu",
        padding='same',
    )(conv_5x5)

    # Pooling
    pool = K.layers.MaxPooling2D(
        pool_size=3,
        strides=1,
        padding='same',
    )(A_prev)

    pool = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        activation="relu",
        padding='same',
    )(pool)

    # Concatenate all the layers
    A = K.layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5, pool])

    return A
