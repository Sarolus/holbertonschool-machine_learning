#!/usr/bin/env python3
"""
    Backward Algorithm for a Hidden Markov Model.
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
        Performs the backward algorithm for a hidden markov model

        Args:
            Observation (list): list of observations
            Emission (dict): dictionary of emission probabilities
            Transition (dict): dictionary of transition probabilities
            Initial (dict): dictionary of initial probabilities

        Returns:
            (dict): dictionary of backward probabilities
    """
    try:
        hidden_states = Emission.shape[0]
        observation_nb = Observation.shape[0]
        backward_path = np.zeros((hidden_states, observation_nb))

        backward_path[:, observation_nb - 1] = np.ones((hidden_states))

        for time in range(observation_nb - 2, -1, -1):
            for state in range(hidden_states):
                transition = Transition[state, :]
                emission = Emission[:, Observation[time + 1]]
                backward_path[state, time] = np.sum(
                    backward_path[:, time + 1] * transition * emission
                )

        probs = np.sum(
            backward_path[:, 0] * Initial[:, 0] * Emission[:, Observation[0]]
        )

        return probs, backward_path
    except Exception:
        return None, None
