#!/usr/bin/env python3
"""
    Viterbi Algorithm for a Hidden Markov Model.
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
        Performs the viterbi algorithm for a hidden markov model

        Args:
            Observation (list): list of observations
            Emission (dict): dictionary of emission probabilities
            Transition (dict): dictionary of transition probabilities
            Initial (dict): dictionary of initial probabilities

        Returns:
            (dict): dictionary of sequences probabilities.
    """
    try:
        hidden_states = Emission.shape[0]
        observation_nb = Observation.shape[0]
        viterbi_sequence = np.zeros((hidden_states, observation_nb))
        backpointer = np.zeros((hidden_states, observation_nb))

        viterbi_sequence[:, 0] = Initial.T * Emission[:, Observation[0]]

        for time in range(1, observation_nb):
            for state in range(hidden_states):
                # Formula :
                # T1[i, j] = max(T1[k, t - 1] * A[k, s] * B[s, y[t]])
                # T2[i, j] = argmax(T1[k, t - 1] * A[k, s] * B[s, y[t]])
                transition = Transition[:, state]
                emission = Emission[state, Observation[time]]
                viterbi_sequence[state, time] = np.max(
                    viterbi_sequence[:, time - 1] * transition * emission
                )
                backpointer[state, time] = np.argmax(
                    viterbi_sequence[:, time - 1] * transition * emission
                )

        # ZT = argmax(T1[k, T - 1])
        current = np.argmax(viterbi_sequence[:, observation_nb - 1])
        best_path = [current]

        # for o in range(T - 1, -1, -1): Backtrack from last observation.
        # best_path.insert(0, S[k]) Insert previous state on most likely path
        # k ← pointers[k, o] Use backpointer to find best previous state
        for time in range(observation_nb - 1, 0, -1):
            best_path.append(int(backpointer[best_path[-1], time]))
        best_path = best_path[::-1]

        probs = np.max(viterbi_sequence[:, observation_nb - 1])

        return best_path, probs
    except Exception as exception:
        return None, None
