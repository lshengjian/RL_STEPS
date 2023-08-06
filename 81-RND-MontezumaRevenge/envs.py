#import gym
import gymnasium  as gym
import numpy as np

from collections import deque
from copy import copy

from torch.multiprocessing import Pipe, Process

from model import *
from PIL import Image

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        #self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, t,info = self.env.step(action)
            # if self.is_render:
            #     self.env.render()
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done,t, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        

class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super().__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def _unwrap(self, env):
        if hasattr(env, "unwrapped"):
            return env.unwrapped
        elif hasattr(env, "env"):
            return unwrap(env.env)
        elif hasattr(env, "leg_env"):
            return unwrap(env.leg_env)
        else:
            return env

    def get_current_room(self):
        ram = self._unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, timeout,info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, timeout,info

    def reset(self):
        return self.env.reset()


class AtariEnvironment(Process):
    def __init__(
            self,
            env_name,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            sticky_action=True,
            p=0.25,
            max_episode_steps=18000):
        super().__init__()
        self.daemon = True
        mode='human' if is_render else 'rgb_array'
        self.env = MaxAndSkipEnv(gym.make(env_name,render_mode=mode))
        if 'Montezuma' in env_name:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_name else 1)
        self.env_name = env_name
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p
        self.max_episode_steps = max_episode_steps

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super().run()
        while True:
            action = self.child_conn.recv()
            force_done=False

            if 'Breakout' in self.env_name:
                action += 1

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
            self.last_action = action

            s, reward, done,_, info = self.env.step(action)

            if self.max_episode_steps < self.steps:
                force_done = True

            log_reward = reward
            #force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)

            self.rall += reward
            self.steps += 1

            if done or force_done:
                self.recent_rlist.append(self.rall)
                if 'Montezuma' in self.env_name:
                    mr= np.mean(self.recent_rlist)
                    vs=info.get('episode', {}).get('visited_rooms', {})
                    print(f"[Episode {self.episode}({self.env_idx})] Step: {self.steps}  Reward: {self.rall}  Recent Reward: {mr}  Visited Room: [{vs}]")
                       
                        
                else:    
                    print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(
                        self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist)))

                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s,_ = self.env.reset()
        #mode='rgb_array'
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        #X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        #x = cv2.resize(X, (self.h, self.w))
        #return x
        frame = Image.fromarray(X).convert('L')
        frame = np.array(frame.resize((self.h, self.w)))
        return frame.astype(np.float32)

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)