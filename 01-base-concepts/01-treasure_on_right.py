"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
"""

import numpy as np
import time

np.random.seed(1234)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = (0,1) #['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.16    # fresh time for one move


def build_q_table(n_states, actions):
    table = np.zeros( (n_states, len(actions) ) )    # q_table initial values
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table[state, :]
    if  ((state_actions == 0).all()) or (np.random.uniform() > EPSILON) :  # act non-greedy or state-action have no value
        action_index = np.random.choice(ACTIONS)
    else:   # act greedy
        action_index = state_actions.argmax() 
    return action_index


def get_env_feedback(s, a):
    done=False
    r = 0
    s_ = s
    # This is how agent will interact with the environment
    if a == 1:    # move right
        if s < N_STATES - 1:
            s_ = s + 1
            if s_ == N_STATES - 1:   # terminate
                r = 1
                done=True
                
    else:   # move left
        if s > 0:
            s_ = s - 1
    return s_, r, done


def update_env(s,done, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if done:
        interaction = f'Episode {episode+1}: total_steps = {step_counter}'
        print(f'\r{interaction}', end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print(f'\r{interaction}', end='')
        time.sleep(FRESH_TIME)


def main():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        cnt = 0
        s = 0
        is_terminated = False
        update_env(s,is_terminated, episode, cnt)
        while not is_terminated:
            a = choose_action(s, q_table)
            s_, r ,is_terminated = get_env_feedback(s, a)  # take action & get next state and reward
            q_predict = q_table[s, a]
            if not is_terminated:
                q_target = r + GAMMA * q_table[s_, :].max()   # next state is not terminal
            else:
                q_target = r     # next state is terminal
       

            q_table[s, a] += ALPHA * (q_target - q_predict)  # update
            s = s_  # move to next state

            update_env(s,is_terminated, episode, cnt+1)
            cnt += 1
    return q_table


if __name__ == "__main__":
    q_table = main()
    print('\r\nQ-table:\n')
    print(q_table)
