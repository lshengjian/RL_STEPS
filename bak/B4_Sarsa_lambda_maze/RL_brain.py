import numpy as np



class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table ={}

    def check_state_exist(self, state):
        if state not in self.q_table:
            # append new state to q table
            row= np.array( \
                    [0]*len(self.actions), \
                    dtype=float 
                )
            self.q_table[state]=row

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_actions = self.q_table[observation]
            action = np.argmax(state_actions)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# backward eligibility traces
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = {}

    def check_state_exist(self, state):
        if state not in self.q_table:
            row= np.array( \
                    [0]*len(self.actions), \
                    dtype=float 
                )
            self.q_table[state]=row
            # also update eligibility trace
            self.eligibility_trace[state]=row.copy()

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table[s_][a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        self.eligibility_trace[s][a] += 1

        # Method 2:
        # self.eligibility_trace.loc[s] *= 0
        # self.eligibility_trace.loc[s][a] = 1

        # Q update
        for k in self.eligibility_trace.keys():
            self.q_table[k] += self.lr * error * self.eligibility_trace[k]

            # decay eligibility trace after update
            self.eligibility_trace[k] *= self.gamma*self.lambda_
