import gymnasium as gym
import numpy as np

from gymnasium.utils.play import play,PlayPlot
'''
https://gymnasium.farama.org/environments/atari/montezuma_revenge/
https://gymnasium.farama.org/api/utils/
'''
NOOP=0
FIRE=1
UP=2
RIGHT=3
LEFT=4
DOWN=5
UPRIGHTFIRE=14
UPLEFTFIRE=15

keys_to_action={
    "j": LEFT ,
    "k": DOWN ,
    "l": RIGHT ,
    "i": UP ,
    "w": FIRE,
    "q": UPLEFTFIRE,
    "e": UPRIGHTFIRE,

}
def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    return [rew,]
def main():
   env = gym.make('MontezumaRevengeNoFrameskip-v4',render_mode="rgb_array",obs_type="grayscale")#obs_type="rgb" #ALE/MontezumaRevenge-v5
   obs,_=env.reset()
   assert obs.shape==(210, 160)
   play(env, keys_to_action=keys_to_action, noop=NOOP)
#    plotter = PlayPlot(callback, 150, ["reward"])
#    play(env, keys_to_action=keys_to_action, noop=NOOP,callback=plotter.callback)

if __name__ == "__main__":
   main()