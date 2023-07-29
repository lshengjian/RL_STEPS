
import numpy as np

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = {}

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_actions = self.q_table[observation]
            action = np.argmax(state_actions)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table[s_].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table[s][a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        #if state not in self.q_table.index:
        
        if not state in self.q_table:
            row= np.array( \
                    [0]*len(self.actions), \
                    dtype=float 
                )
            self.q_table[state]=row
