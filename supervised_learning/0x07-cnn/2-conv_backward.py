#!/usr/bin/env python3
"""
    Performs back propagation over a convolutional layer
    of a neural network
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
        Performs back propagation over a convolutional layer
        of a neural network

        Args:
            dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
                containing the gradient of the cost with respect to the output
                of the convolutional layer
            A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
            W: numpy.ndarray of shape (f, f, c_prev, c_new)
                containing the kernels for the convolution
            b: numpy.ndarray of shape (1, 1, 1, c_new)
                containing the biases applied to the convolution
            padding: string that is either same or valid, indicating
                     the type of padding used
            stride: tuple of (sh, sw) containing the strides for
                    the convolution

        Returns:
            input_derivative: numpy.ndarray containing the gradient of the
                              cost with respect to the previous layer
            filter_derivative: numpy.ndarray containing the gradient
                               of the cost with respect to W
            db: numpy.ndarray containing the gradient of the cost
                with respect to b
    """
    output_nb, output_height, output_width, output_channels = dZ.shape
    _, image_height, image_width, _ = A_prev.shape
    filter_height, filter_width, _, _ = W.shape
    stride_height, stride_width = stride

    if padding == 'same':
        padding_height = int(
            np.ceil(((image_height - 1) * stride_height +
                    filter_height - image_height) / 2)
        )
        padding_width = int(
            np.ceil(((image_width - 1) * stride_width +
                    filter_width - image_width) / 2)
        )
    elif padding == 'valid':
        padding_height = 0
        padding_width = 0

    A_prev = np.pad(A_prev, ((0, 0), (padding_height, padding_height),
                             (padding_width, padding_width), (0, 0)),
                    'constant')

    input_derivative = np.zeros(A_prev.shape)
    filter_derivative = np.zeros(W.shape)
    bias_derivative = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for output_index in range(output_nb):
        for row in range(output_height):
            for column in range(output_width):
                for channel in range(output_channels):
                    x = row * stride_height
                    y = column * stride_width
                    filter = W[:, :, :, channel]
                    output_derivative = dZ[output_index, row, column, channel]
                    slice_a = A_prev[output_index, x:x +
                                     filter_height, y:y+filter_width, :]
                    filter_derivative[:, :, :,
                                      channel] += slice_a * output_derivative
                    input_derivative[output_index, x:x+filter_height,
                                     y:y+filter_width,
                                     :] += output_derivative * filter
    if padding == 'same':
        input_derivative = input_derivative[:,
                                            padding_height:-padding_height,
                                            padding_width:-padding_width, :]
    return input_derivative, filter_derivative, bias_derivative
