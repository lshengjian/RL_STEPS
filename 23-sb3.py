from rlbase.policies.manual import ManualPolicy
from rlbase.plugins.renderer_policy import PolicyRenderer
from stable_baselines3 import PPO,DQN
from rlbase import MyGrid
from rlbase.core import ACTION_FLAGS
import hydra

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    
    # env=MyGrid(None,cfg.map_4x4,cfg.render.fps,cfg.render.win_size) #
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=100_000)
    # model.save('sb3-ppo.zip')
    # env.close()
    env = MyGrid('human',cfg.map_4x4,cfg.render.fps,cfg.render.win_size)
    env.game.add_plugin(PolicyRenderer(env.game.model,3,env.renderer))
    model=PPO.load('sb3-ppo.zip')
    
    obs,_=env.reset()
    mgr=ManualPolicy(env)
    

    while mgr.running:
        s=env.model.state
        action=mgr.decition(s) # for quit
        
        action, _ = model.predict(obs, deterministic=True)
        
        pi=env.model.ext_data['pi'][s]
        pi*=0
        pi[action]=1
        obs, r,done,*_ = env.step(action)
        print(obs,ACTION_FLAGS[action], r,done)
        if done:
            obs,_=env.reset()
        


    env.close()

if __name__ == '__main__':
    main()