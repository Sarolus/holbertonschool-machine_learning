#!/usr/bin/env python3
"""
    Forward Algorithm for a Hidden Markov Model.
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
        Performs the forward algorithm for a hidden markov model

        Args:
            Observation (list): list of observations
            Emission (dict): dictionary of emission probabilities
            Transition (dict): dictionary of transition probabilities
            Initial (dict): dictionary of initial probabilities

        Returns:
            (dict): dictionary of forward probabilities
    """
    try:
        hidden_states = Emission.shape[0]
        observation_nb = Observation.shape[0]
        forward_path = np.zeros((hidden_states, observation_nb))

        forward_path[:, 0] = Initial.T * Emission[:, Observation[0]]

        for time in range(1, observation_nb):
            for state in range(hidden_states):
                forward_path[state, time] = np.sum(
                    np.dot(
                        np.dot(
                            forward_path[:, time - 1],
                            Transition[:, state]
                        ),
                        Emission[state, Observation[time]]
                    )
                )

        probs = np.sum(forward_path[:, observation_nb - 1])

        return probs, forward_path
    except Exception:
        return None, None
