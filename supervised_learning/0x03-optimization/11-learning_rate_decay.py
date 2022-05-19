#!/usr/bin/env python3
"""
    Learning Rate Decay Module
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        Creates the operation to perform the learning rate decay
    """

    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
