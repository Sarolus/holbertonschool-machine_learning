#!/usr/bin/env python3
"""
    Convolution Forward Propagation Module
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
        Performs a forward propagation over a convolutional layer of a neural
        network.

        Args:
            A_prev (np.ndarray): np.ndarray containing the output of the
            previous layer.
            W (np.ndarray): np.ndarray containing the kernels for the
            convolution.
            b (np.ndarray): np.ndarray containing the biases applied to the
            convolution.
            activation: The activation function applied to the convolution.
            padding (str, optional): Indicator of the type of padding used.
            Defaults to "same".
            stride (tuple, optional): Tuple containing the strides for the
            convolution. Defaults to (1, 1).

        Raises:
            ValueError: Padding must be valid or same

        Returns:
            np.ndarray: The output of the convolutional layer.
    """
    input_nb, input_height, input_width, input_channel = A_prev.shape
    filter_height, filter_width, input_channel, output_channel = W.shape
    stride_height, stride_width = stride

    if padding == "same":
        padding_height = int(
            (
                (input_height - 1) * stride_height +
                filter_height - input_height
            ) / 2
        )
        padding_width = int(
            (
                (input_width - 1) * stride_width +
                filter_width - input_width
            ) / 2
        )
    elif padding == "valid":
        padding_height = 0
        padding_width = 0
    else:
        raise ValueError("padding must be valid or same")

    output_height = int(
        (input_height + 2 * padding_height - filter_height) / stride_height
    ) + 1
    output_width = int(
        (input_width + 2 * padding_width - filter_width) / stride_width
    ) + 1

    output = np.zeros((input_nb, output_height, output_width, output_channel))

    padded_input = np.pad(
        A_prev,
        (
            (0, 0),
            (padding_height, padding_height),
            (padding_width, padding_width),
            (0, 0)
        ),
        mode="constant"
    )

    kernels_copy = W.copy()
    for row in range(output_height):
        for column in range(output_width):
            x = row * stride_height
            y = column * stride_width
            images_slide = padded_input[
                :,
                x:x + filter_height,
                y:y + filter_width,
                :
            ]

            for kernel_index in range(output_channel):
                output[:, row, column, kernel_index] = np.tensordot(
                    images_slide,
                    kernels_copy[:, :, :, kernel_index],
                    axes=3
                ) + b[:, :, :, kernel_index]

    if activation is None:
        return output

    return activation(output)
