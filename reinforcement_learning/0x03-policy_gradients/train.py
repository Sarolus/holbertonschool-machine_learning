#!/usr/bin/env python3
"""
    Full Training with Policy Gradient Implementation Module
"""

import gym
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
        Function that implements a full training.

        Args:
            env (env): Initial environment
            nb_episodes (int): number of episodes used for training.
            alpha (float, optional): The learning rate. Defaults to 0.000045.
            gamma (float, optional): The discount factor. Defaults to 0.98.
            show_result (bool, optional): Result rendering parameter.
            Defaults to False.
    """
    w = np.random.rand(4, 2)
    eps_rewards = []

    for e in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0
        while True:
            if show_result and (e % 1000 == 0):
                env.render()
            action, grad = policy_gradient(state, w)
            new_state, reward, done, _ = env.step(action)
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = new_state[None, :]
            if done:
                break
        for i in range(len(grads)):
            w += (alpha * grads[i] *
                  sum([r * gamma**r for t, r in enumerate(rewards[i:])]))
        eps_rewards.append(score)
        print("{}: {}".format(e, score), end="\r", flush=False)

    return eps_rewards
