import gymnasium as gym
import numpy as np

from gymnasium.utils.play import play,PlayPlot
'''
https://gymnasium.farama.org/api/vector/
'''

def make_env():
   return gym.make("MontezumaRevengeNoFrameskip-v4", render_mode="rgb_array",obs_type="rgb")
   

def main():
   num_envs=5
   env = gym.vector.AsyncVectorEnv([ make_env ]*num_envs)
   obs,_=env.reset()
   assert obs.shape==(5,210, 160,3)


if __name__ == "__main__":
   main()