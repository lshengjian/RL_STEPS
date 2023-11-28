from rlbase import MiniGrid,MyGrid
from rlbase.policies.manual import ManualPolicy
from rlbase.plugins.renderer_return import ReturnRenderer
import hydra

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    #env=MiniGrid('human',cfg.map_4x4,cfg.render.fps,cfg.render.win_size)
    env=MyGrid('human',cfg.map_4x4,cfg.render.fps,cfg.render.win_size)
    env.game.add_plugin(ReturnRenderer(env.game.model,2,env.renderer,cfg.algorithm.gamma))
    policy=ManualPolicy(env)
   
    s,_=env.reset(seed=42)
    
    while policy.running:
        action=policy.decition(s)
        s,r,done,*_=env.step(action)
        print(s,r)
        if done:
            break

    env.close()

if __name__ == "__main__":
    main()

    