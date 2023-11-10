import gymnasium as gym
import numpy as np

from gymnasium.wrappers import AtariPreprocessing
'''
https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing
'''

def make_env():
   env = gym.make("MontezumaRevengeNoFrameskip-v4", render_mode="rgb_array",obs_type="rgb")
   return AtariPreprocessing(env,scale_obs=True)
   

def main():
   num_envs=5
   env = gym.vector.AsyncVectorEnv([ make_env ]*num_envs,shared_memory=True)
   obs,_=env.reset()
   assert obs.shape==(5,84, 84)
   assert (obs[0]<1).all()
   print(obs[0][40][40:50])
   obs,rs,*_=env.step([0]*num_envs)
   assert obs.shape==(5,84, 84)
   assert rs.shape==(5,)
   assert (rs==0).all()


if __name__ == "__main__":
   main()