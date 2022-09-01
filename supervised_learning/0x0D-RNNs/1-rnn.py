#!/usr/bin/env python3
"""
    Forward propagation for a RNN
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
        Performs forward propagation for a RNN

        Args:
            rnn_cell: instance of RNNCell
            X: data to propagate in the form of a numpy.ndarray
               of shape (t, m, i)
                t: the number of time steps
            h_0: initial hidden state in the form of a numpy.ndarray
                 of shape (m, h)
                m: the batch size
                h: the dimension of the hidden state

        Returns: H, Y
            H: numpy.ndarray containing all of the hidden states
                shape (t, m, h)
            Y: numpy.ndarray containing all of the outputs
                shape (t, m, o)
    """

    t, m, i = X.shape
    h = h_0.shape[1]
    H = np.zeros((t, m, h))
    Y = np.zeros((t, m, rnn_cell.o))
    H[0] = h_0

    for i in range(t):
        H[i], Y[i] = rnn_cell.forward(H[i - 1], X[i])

    return H, Y
