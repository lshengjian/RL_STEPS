from rlbase import MiniGrid
from rlbase.policies.manual import ManualPolicy
from rlbase.policies.model.policy_iteration import IterationPolicy as Policy
from rlbase.plugins.renderer_policy import PolicyRenderer

import hydra

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    env=MiniGrid('human',cfg.map_8x8,cfg.render.fps,cfg.render.win_size)
    pr=PolicyRenderer(env.game.model,3,env.renderer)
    env.game.add_plugin(pr)
    policy=Policy(env,cfg.algorithm.gamma)
    policy.truncated_policy_iteration()
    s,_=env.reset(seed=123)
    pr._model.ext_data['pi']=policy.pi
    policy=ManualPolicy(env)
    while policy.running:
        action=policy.decition(s)
        s1,r,t,*_=env.step(action)
        if t :
            break
    env.close()


if __name__ == "__main__":
    main()