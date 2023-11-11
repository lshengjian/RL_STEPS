from ..data import G, NUM_ACTIONS, Tile
import numpy as np
import esper
import copy


class PolicySystem(esper.Processor):
    def __init__(self, P: np.ndarray, gamma: float = 0.9):
        super().__init__()
        row, col = G.GRID_SIZE
        total = row*col
        self.nS = total
        self.nA = NUM_ACTIONS
        self.gamma = gamma
        self.P = P
        self.minx = 0
        self.maxx = col-1
        self.miny = 0
        self.maxy = row-1
        self.pi = np.ones((total, NUM_ACTIONS), dtype=float)/NUM_ACTIONS
        self.V = np.zeros(total, dtype=float)
        self.Q = np.zeros((total, NUM_ACTIONS), dtype=float)

    def q_from_v(self, s: int):
        gamma = self.gamma
        q = np.zeros(self.nA)
        for a in range(self.nA):
            for prob, next_state, reward, _ in self.P[s][a]:
                q[a] += prob * (reward + gamma * self.V[next_state])
        return q

    def truncated_policy_evaluation(self, max_it: int = 3):
        
        num_it = 0
        while num_it < max_it:
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

            # OPTION 1: construct a deterministic policy
            # policy[s][np.argmax(q)] = 1

            # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
            best_a = np.argwhere(q == np.max(q)).flatten()
            # print(s,best_a)
            policy[s] = np.sum([np.eye(self.nA)[i]
                               for i in best_a], axis=0)/len(best_a)

        return policy

    def truncated_policy_iteration(self, max_it=5, eps=1e-8):
        while True:
            policy = self.policy_improvement()
            old_V = copy.copy(self.V)
            self.truncated_policy_evaluation(max_it)
            if max(abs(self.V-old_V)) < eps:
                break

    def make_policy(self, repeats, eps):
        self.truncated_policy_iteration(repeats, eps)

    def process(self):
        self.make_policy( 10, 1e-8)
        # This will iterate over every Entity that has Tile  component:
        for _, tile in esper.get_component(Tile):
            state = tile.state
            # self.V[state]= np.sum(self.PI[state]*self.Q[state])
