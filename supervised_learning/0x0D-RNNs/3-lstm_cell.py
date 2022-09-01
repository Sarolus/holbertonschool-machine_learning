#!/usr/bin/env python3
"""
    LSTM Cell
"""

import numpy as np


class LSTMCell:
    """
        Represents an LSTM unit

        Attributes:
            Wf: for the forget gate
            Wu: for the update gate
            Wc: for the cell gate
            Wo: for the output gate
            Wy: for the output
            bf: for the forget gate
            bu: for the update gate
            bc: for the cell gate
            bo: for the output gate
            by: for the output
    """

    def __init__(self, i, h, o):
        """
            Constructor

            Args:
                i: dimension of the input
                h: dimension of the hidden state
                o: dimension of the output
        """
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.i = i
        self.h = h
        self.o = o

    def forward(self, h_prev, c_prev, x_t):
        """
            Performs forward propagation fo one time step

            Args:
                h_prev: previous hidden state
                c_prev: previous cell state
                x_t: input

            Returns:
                h_next: next hidden state
                c_next: next cell state
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        concat = np.concatenate((h_prev, x_t), axis=1)
        f = sigmoid(np.dot(concat, self.Wf) + self.bf)
        i = sigmoid(np.dot(concat, self.Wu) + self.bu)
        c_next = f * c_prev + i * np.tanh(np.dot(concat, self.Wc) + self.bc)
        o = sigmoid(np.dot(concat, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)
        y = np.dot(h_next, self.Wy) + self.by
        y = softmax(y)

        return h_next, c_next, y
