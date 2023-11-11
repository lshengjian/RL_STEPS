from mygrid.env  import MiniGrid
from mygrid.config import ACT_NAMES
from mygrid.utils import greedy_select
from random import random
import gymnasium as gym
import numpy as np
import time,copy
import hydra

def q_from_v(env:MiniGrid, V:np.ndarray, s:int, gamma:float=0.9):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.world.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

def policy_improvement(env:MiniGrid, V:np.ndarray,  gamma:float=0.9):
    policy = np.ones([env.nS, env.nA]) / env.nA 
    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)
        
        # OPTION 1: construct a deterministic policy 
        # policy[s][np.argmax(q)] = 1
        
        # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
        best_a = np.argwhere(q==np.max(q)).flatten()
        #print(s,best_a)
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)
        
    return policy

def truncated_policy_evaluation(env:MiniGrid, policy:np.ndarray, V:np.ndarray, \
                                max_it:int=3, gamma:float=0.9):
    num_it=0
    while num_it < max_it:
        for s in range(env.nS):
            v = 0
            q = q_from_v(env, V, s, gamma)
            for a, action_prob in enumerate(policy[s]):
                v += action_prob * q[a]
            V[s] = v
        num_it += 1
    return V

def truncated_policy_iteration(env, max_it=5, gamma=1, eps=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        policy = policy_improvement(env, V)
        old_V = copy.copy(V)
        V = truncated_policy_evaluation(env, policy, V, max_it, gamma)
        if max(abs(V-old_V)) < eps:
            break
    return policy, V


def make_policy(env:MiniGrid,cfg):
   policy, V=truncated_policy_iteration(env,
        cfg.optim.repeats,
        cfg.optim.gamma,
        cfg.optim.eps)
   env.V=V
   env.PI=policy

@hydra.main(version_base="1.1", config_path=".", config_name="02-config")
def main(cfg: "DictConfig"):  # noqa: F821
   env=MiniGrid(render_mode="human",
                map_name=cfg.env.map_name, 
                fps=cfg.env.fps,
                is_terminate_reach_goal=True)
   s,_=env.reset()
   make_policy(env,cfg)
  
   for _ in range(cfg.simulate.steps):
      pi=env.PI[s]
      a=greedy_select(pi,env.np_random)
    #   r,c=env.world.state2idx(s)
    #   print(f'({r+1},{c+1})| {ACT_NAMES[a]}')
      s,  r, terminated, truncated, _  = env.step(a)
      if terminated or truncated:
         time.sleep(0.5)
         s,_=env.reset()
         #make_policy(env)

   env.close()
   #plot_values(env.V)


if __name__ == "__main__":
   main()
   