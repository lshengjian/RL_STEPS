from rlbase import MiniGrid
from rlbase.policies.manual import ManualPolicy

from rlbase.plugins.renderer_visited import VisitedRenderer
from rlbase.plugins.renderer_policy import PolicyRenderer
import hydra

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    env=MiniGrid('human',cfg.map_4x4,cfg.render.fps,cfg.render.win_size)
    env.game.add_plugin(VisitedRenderer(env.game.model,2,env.renderer))
    env.game.add_plugin(PolicyRenderer(env.game.model,3,env.renderer))
    policy=ManualPolicy(env)
   
    s,_=env.reset(seed=42)
    
    for _ in range(1000):
        action=policy.decition(s)
        s,*_=env.step(action)

    env.close()

if __name__ == "__main__":
    main()