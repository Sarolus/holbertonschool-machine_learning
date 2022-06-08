#!/usr/bin/env python3
"""
    Convolutional Backward Propagation Module
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

    # Padding calculate
    # Calculate the padding height and width
    if padding == 'same':
        padding_height = int(
            (
                (
                    (image_height - 1) * stride_height +
                    filter_height - image_height
                ) / 2
            ) + 1
        )
        padding_width = int(
            (
                (
                    (image_width - 1) * stride_width +
                    filter_width - image_width
                ) / 2
            ) + 1
        )
    elif padding == 'valid':
        padding_height = 0
        padding_width = 0
    else:
        raise NameError('padding must be same, valid, or a tuple')

    # Initialize dA_prev
    input_derivative = np.zeros(A_prev.shape)

    # Initialize dW
    filter_derivative = np.zeros(W.shape)

    # Calculate db
    bias_derivative = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Loop over outputs
    for output_index in range(output_nb):
        # Loop over vertical axis
        for row in range(output_height):
            # Loop over horizontal axis
            for column in range(output_width):
                x = column * stride_width
                y = row * stride_height
                slice = A_prev[
                    output_index,
                    x:x + filter_height,
                    y:y + filter_width,
                    :
                ]

                # Loop over channels
                for channel in range(output_channels):
                    filter = W[:, :, :, channel]
                    output_derivative = dZ[output_index, row, column, channel]

                    # Apply the slice to the filter
                    filter_derivative[:, :, :,
                                      channel] += slice * output_derivative

                    # Apply the filter to the input
                    input_derivative[
                        output_index,
                        x:x + filter_height,
                        y:y + filter_width,
                        :
                    ] += output_derivative * filter

        # If padding is same apply it
        if padding == 'same':
            dA_prev = dA_prev[
                :,
                padding_height:-padding_height,
                padding_width:-padding_width,
                :
            ]

    return input_derivative, filter_derivative, bias_derivative
