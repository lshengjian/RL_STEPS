import gymnasium as gym
import numpy as np
from gymnasium.utils.play import play
from myenv.myenv  import MyLakeEnv

from myenv.config import *

keys_to_action={
    "a": LEFT ,
    "s": DOWN ,
    "d": RIGHT ,
    "w": UP ,
    "z": STAY
}
def main():
   env = MyLakeEnv(render_mode="rgb_array",map_name="8x8",is_slippery=False)
   play(env, keys_to_action=keys_to_action, noop=STAY)

if __name__ == "__main__":
   main()