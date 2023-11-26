import gymnasium as gym

from stable_baselines3 import PPO
from rlbase import MiniGrid
import hydra

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    env=MiniGrid(None,cfg.map_5x5,cfg.render.fps,cfg.render.win_size) #
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20_000)

    
    env2 = MiniGrid('human',cfg.map_5x5,cfg.render.fps,cfg.render.win_size) #
    obs,_=env2.reset()
    for _ in range(100):
        actions, _states = model.predict([obs], deterministic=True)
        obs, r,done,*_ = env2.step(actions[0])
        if done:
            obs,_=env2.reset()
        


    env.close()

if __name__ == '__main__':
    main()