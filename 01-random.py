
from rlbase import MiniGrid
from rlbase.policies.random import RandomPolicy
import hydra

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    env=MiniGrid('human',cfg.map_8x8,cfg.render.fps,cfg.render.win_size)
    state,_=env.reset(seed=42)
    policy=RandomPolicy(env)
    for _ in range(30):
        action=policy.decition(state)
        state,*_=env.step(action)

    env.close()

if __name__ == "__main__":
    main()
