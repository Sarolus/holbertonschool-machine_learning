#!/usr/bin/env python3
"""
    Script that performs the Baum-Welch algorithm for a hidden markov model.
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


def get_xi(alpha, transition, beta, emission, observations):
    """
        Calculates the expected number of transitions between states.

        Args:
            alpha: forward probabilities
            transition: transition probabilities
            beta: backward probabilities
            emission: emission probabilities
            observations: list of observations

        Returns:
            xi: expected number of transitions between states
    """

    t = observations.shape[0]
    m, _ = emission.shape
    xi = np.zeros((m, m, t - 1))

    for time in range(t - 1):
        numerator = np.dot((np.dot(alpha[:, time].T, transition) *
                            emission[:, observations[time + 1]].T),
                           beta[:, time + 1])
        for i in range(m):
            denominator = (alpha[i, time] * transition[i] *
                           emission[:, observations[time + 1]].T *
                           beta[:, time + 1].T)
            xi[i, :, time] = denominator / numerator
    return xi


def get_transition(xi, gamma_tmp):
    """
        Calculates the expected number of transitions between states.

        Args:
            xi: expected number of transitions between states
            gamma_tmp: number of times each state was visited

        Returns:
            transition: expected number of transitions between states
    """

    return np.sum(xi, 2) / np.sum(gamma_tmp, axis=1).reshape((-1, 1))


def get_emission(gamma, observations, emission, n):
    """
        Calculates the expected number of transitions between states.

        Args:
            gamma: number of times each state was visited
            observations: list of observations
            emission: emission probabilities
            n: number of observations

        Returns:
            emission: expected number of transitions between states
    """

    for ni in range(n):
        emission[:, ni] = np.sum(gamma[:, observations == ni],
                                 axis=1)
    emission /= np.sum(gamma, axis=1).reshape((-1, 1))

    return emission


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
        Performs the Baum-Welch algorithm for a hidden markov model

        Args:
            Observations: list of observations
            Transition: transition matrix
            Emission: emission matrix
            Initial: initial probabilities
            iterations: number of iterations to perform

        Returns:
            Transition:
            Emision:
    """

    try:
        if not isinstance(Observations, np.ndarray):
            raise TypeError("Observations must be a numpy.ndarray")

        if Observations.ndim != 1:
            raise TypeError("Observations must be a 1D array")

        t = Observations.shape[0]

        if not isinstance(Emission, np.ndarray):
            raise TypeError("Emission must be a numpy.ndarray")

        if Emission.ndim != 2:
            raise TypeError("Emission is not a 2D array")

        m, n = Emission.shape

        if not isinstance(Transition, np.ndarray):
            raise TypeError("Transition must be a numpy.ndarray")

        if Transition.shape != (m, m):
            raise ValueError("Transition matrix has wrong shape")

        if not isinstance(Initial, np.ndarray):
            raise TypeError("Initial must be a numpy.ndarray")

        if Initial.shape != (m, 1):
            raise ValueError("Initial matrix has wrong shape")

        t = Observations.shape[0]
        m, n = Emission.shape

        for _ in range(iterations):
            _, alpha = forward(Observations, Emission, Transition, Initial)
            _, beta = backward(Observations, Emission, Transition, Initial)
            xi = get_xi(alpha, Transition, beta, Emission, Observations)
            gamma_tmp = np.sum(xi, axis=1)
            Transition = np.sum(xi, 2) / np.sum(gamma_tmp,
                                                axis=1).reshape((-1, 1))
            gammas = np.hstack(
                (gamma_tmp, np.sum(xi[:, :, t - 2], axis=0).reshape((-1, 1))))
            emission = get_emission(gammas, Observations, Emission, n)

        return Transition, emission
    except Exception as e:
        return None, None
