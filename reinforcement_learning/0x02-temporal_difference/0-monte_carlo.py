#!/usr/bin/env python3
"""Program that performs the Monte Carlo algorithm"""
import numpy as np
import gym


def episode_gen(env, policy, max_steps):
    """Function that generates an episode"""
    episode = [[], []]
    state = env.reset()
    for i in range(max_steps):
        action = policy(state)
        new_state, reward, done, info = env.step(action)
        episode[0].append(state)
        if env.desc.reshape(env.observation_space.n)[new_state] == b'H':
            episode[1].append(-1)
            return episode
        if env.desc.reshape(env.observation_space.n)[new_state] == b'G':
            episode[1].append(1)
            return episode
        episode[1].append(reward)
        state = new_state
    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Function that performs the Monte Carlo algorithm"""
    discounts = [gamma ** i for i in range(max_steps)]
    for ep in range(episodes):
        episode = episode_gen(env, policy, max_steps)
        for i in range(len(episode[0])):
            Gt = sum(np.array(episode[1][i:]) *
                     np.array(discounts[:len(episode[1][i:])]))
            V[episode[0][i]] = V[episode[0][i]] +\
                alpha * (Gt - V[episode[0][i]])
    return V
