#!/usr/bin/env python3
"""Program that performs the TD(λ) algorithm"""
import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Function that performs the TD(λ) algorithm"""
    Et = [0 for i in range(env.observation_space.n)]
    for i in range(episodes):
        state = env.reset()
        for j in range(max_steps):
            Et = list(np.array(Et) * lambtha * gamma)
            Et[state] += 1
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            if env.desc.reshape(env.observation_space.n)[new_state] == b'H':
                reward = -1
            if env.desc.reshape(env.observation_space.n)[new_state] == b'G':
                reward = 1
            delta = reward + gamma * V[new_state] - V[state]
            V[state] = V[state] + alpha * delta * Et[state]
            if done:
                break
            state = new_state
    return V
