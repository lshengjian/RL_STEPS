import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing,FrameStack

from copy import copy
from config import *

train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])

class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super().__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = self.env.unwrapped.ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done,truc, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done:
            self.visited_rooms.clear()
        return obs, rew, done,truc, info

    def reset(self):
        return self.env.reset()

def make_env(render_mode="rgb_array"):
   env = gym.make("MontezumaRevengeNoFrameskip-v4", \
                max_episode_steps=max_step_per_episode , \
                render_mode=render_mode, \
                obs_type="rgb")
   env = MontezumaInfoWrapper(env,room_address=3)
   env = AtariPreprocessing(env,scale_obs=True)
   env = FrameStack(env,4)
   return env

def get_envs(num_envs=5):
    envs = gym.vector.AsyncVectorEnv([ make_env ]*num_envs,shared_memory=True)
    return envs

def check():
   num_envs=2
   envs = get_envs(num_envs)
   obs,_=envs.reset()
   assert obs.shape==(num_envs,4,84, 84)
   assert (obs[0][0]<1).all()
   print(obs[0][0][40][40:50])
   obs,rs,*_=envs.step([0]*num_envs)
   assert obs.shape==(num_envs,4,84, 84)
   assert rs.shape==(num_envs,)
   assert (rs==0).all()
   envs.close()


if __name__ == "__main__":
   check()