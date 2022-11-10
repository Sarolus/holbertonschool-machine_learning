#!/usr/bin/env python3
"""Program that performs SARSA(λ)"""
import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Function that uses epsilon greedy to determine next action"""
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, int(Q.shape[1]))
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Function that performs SARSA(λ)"""
    init_epsilon = epsilon
    Et = np.zeros((Q.shape))
    for i in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon=epsilon)
        for j in range(max_steps):
            Et = Et * lambtha * gamma
            Et[state, action] += 1.0
            new_state, reward, done, info = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon=epsilon)
            if env.desc.reshape(env.observation_space.n)[new_state] == b'H':
                reward = -1
            if env.desc.reshape(env.observation_space.n)[new_state] == b'G':
                reward = 1
            deltat = reward + gamma * Q[new_state, new_action] -\
                Q[state, action]
            Q[state, action] = Q[state, action] + alpha * deltat *\
                Et[state, action]
            if done:
                break
            state = new_state
            action = new_action
        epsilon = (min_epsilon + (init_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * i))
    return Q
