"""Reinforcement Learning Example1: Basic Q Learning
Visualize Q Learning that through the agent to find the treasure.
Our environment "_o___T", "o": the agent, "T": treasure
The initial position is in the left of 1 dimensional world

Key points
1. Update Q table
2. Decide the action by Q table
"""

import numpy as np
import pandas as pd
import time

STATES_SIZE = 6  # the length of the 1 dimensional world
ACTION = ["left", "right"]  # available actions
EPSILON = 0.9  # greedy police
MAX_ITERATION = 15
GAMMA = 0.9  # discount factor
ALPHA = 0.1  # learning rate
FRESH_TIME = 0.3

np.random.seed(2)

# define q-table
def build_q_table(states_size, action):
    table = pd.DataFrame(np.zeros((states_size, len(action))), columns=action)
    return table

# define how to choose an action
def choose_action(states, q_table):
    states_action = q_table.iloc[states, :]
    if (np.random.rand() > EPSILON) or (states_action.all() == 0):
        action = np.random.choice(list(q_table))
    else:
        action = np.argmax(states_action)
    return action  # return column name

# the agent will interact with the environment
def get_env_feedback(S, A):
    if A == "right":
        if S == STATES_SIZE - 2:
            S_ = "Terminal"
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

# define the environment to visualize the process
def env(S, iteration, step_counter):
    env_list = ["_"] * (STATES_SIZE - 1) + ["T"]
    if S == "Terminal":
        interaction = "Iteration: %s Total_steps = %s" % (iteration, step_counter)
        print("\r{}".format(interaction))
    else:
        env_list[S] = "o"
        interaction = "".join(env_list)
        print("\r{}".format(interaction), end="")
        # \r meaning is to cover the original line
        # end="" meaning to avoid change line
        time.sleep(FRESH_TIME)

# main part of RL
def RL():
    q_table = build_q_table(STATES_SIZE, ACTION)
    for iteration in range(MAX_ITERATION):
        step_counter = 0

        S = 0  # initial position
        is_terminated = False
        env(S, iteration, step_counter + 1)
        while not is_terminated:

            A = choose_action(S, q_table)
            q_est = q_table.loc[S, A]
            S_, R = get_env_feedback(S, A)  # take action, get reward and next state
            if S_ != "Terminal":
                q_real = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_real = R
                is_terminated = True

            # print(q_real)
            q_table.loc[S, A] += ALPHA * (q_real - q_est)
            # print(S, S_)
            # print(q_table)
            S = S_
            env(S, iteration, step_counter + 1)
            step_counter += 1
    return q_table

q_table = RL()
print(q_table)
