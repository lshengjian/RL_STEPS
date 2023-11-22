
from rlbase.core import TState
from copy import deepcopy
from rlbase.core.utils import greedy_select
from gymnasium import Env
from ...envs import MiniGrid
from ..random import RandomPolicy
import numpy as np
class IterationPolicy(RandomPolicy):
    def __init__(self, env: MiniGrid,gamma=0.9):
        super().__init__(env)
        self.nS = env.model.nS
        self.nA = env.model.nA
        self.gamma = gamma
        self.P = env.model.P
        self.pi = np.ones((self.nS, self.nA), dtype=float)/self.nA
        self.V = np.zeros(self.nS, dtype=float)


    def decition(self, state:TState):
        return greedy_select(self.pi[state], self.env.np_random)

    def q_from_v(self, s: int):
        gamma = self.gamma
        q = np.zeros(self.nA)
        for a in range(self.nA):
            for prob, next_state, reward, _ in self.P[s][a]:
                q[a] += prob * (reward + gamma * self.V[next_state])
        return q

    def truncated_policy_evaluation(self, max_it: int = 3):
        for num_it in range(max_it):
            for s in range(self.nS):
                v = 0
                q = self.q_from_v(s)
                for a, action_prob in enumerate(self.pi[s]):
                    v += action_prob * q[a]
                self.V[s] = v
            num_it += 1

    def policy_improvement(self):
        policy = self.pi
        for s in range(self.nS):
            q = self.q_from_v(s)
            policy[s]*=0
            # OPTION 1: construct a deterministic policy
            policy[s,np.argmax(q)] = 1

            # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
            #best_a = np.argwhere(q == np.max(q)).flatten()
            # print(s,best_a)
            #policy[s] = np.sum([np.eye(self.nA)[i]
            #                   for i in best_a], axis=0)/len(best_a)

    def truncated_policy_iteration(self, max_it=5, eps=1e-8):
        for _ in range(100):
            self.policy_improvement()
            old_V =deepcopy(self.V)
            self.truncated_policy_evaluation(max_it)
            delta=max(abs(self.V-old_V)) 
            if delta < eps:
                break


