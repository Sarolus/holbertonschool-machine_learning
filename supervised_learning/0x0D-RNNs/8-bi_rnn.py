#!/usr/bin/env python3
"""
    Bidirectional RNN
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
        Performs forward propagation for a bidirectional RNN

        Args:
            bi_cell: instance of the BidirectionalCell class
            X: data to propagate through the RNN
            h_0: array containing the initial hidden state for each
                RNN cell
            h_t: array containing the final hidden state for each RNN cell

        Returns:
            H: array containing all of the hidden states of the RNN
            Y: array containing all of the outputs of the RNN
    """

    t, m, _ = X.shape
    _, h = h_0.shape

    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))

    for iteration in range(t):
        if iteration == 0:
            h_prev, h_next = h_0, h_t
        else:
            h_prev, h_next = Hf[iteration - 1], Hb[t - iteration]

        Hf[iteration] = bi_cell.forward(h_prev, X[iteration])
        Hb[t - 1 - iteration] = bi_cell.backward(h_next, X[t - 1 - iteration])

    H = np.concatenate((Hf, Hb), axis=2)

    Y = bi_cell.output(H)

    return H, Y
