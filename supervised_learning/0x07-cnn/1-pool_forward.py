#!/usr/bin/env python3
"""
    Pooling Forward Propagation Module
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        Performs forward propagation over a pooling layer of a neural
        network.

        Args:
            A_prev (np.ndarray): np.ndarray containing the output of the
            previous layer.
            kernel_shape (tuple): tuple containing the size of the kernel
            for the pooling.
            stride (tuple, optional): tuple containing the strides for the
            pooling. Defaults to (1, 1).
            mode (str, optional): Indicator to perform whether maximum or
            average pooling. Defaults to 'max'.

        Returns:
            np.ndarray: Output of the pooling layer.
    """
    input_nb, input_height, input_width, input_channel = A_prev.shape
    filter_height, filter_width = kernel_shape
    stride_height, stride_width = stride

    output_height = int(
        (
            (input_height - filter_height) / stride_height
        ) + 1
    )
    output_width = int(
        (
            (input_width - filter_width) / stride_width
        ) + 1
    )

    output = np.zeros((input_nb, output_height, output_width, input_channel))

    pooling = np.average if mode == 'avg' else np.max

    for column in range(output_width):
        for row in range(output_height):
            output[:, row, column, :] = pooling(
                A_prev[
                    :,
                    row * stride_height: row *
                    stride_height + filter_height,
                    column * stride_width: column *
                    stride_width + filter_width,
                    :
                ],
                axis=(1, 2),
            )

    return output
