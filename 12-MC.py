from rlbase import MiniGrid
from rlbase.core import Transition
from rlbase.core.utils import categorical_sample
from rlbase.policies.manual import ManualPolicy as Policy
from rlbase.plugins.renderer_policy import PolicyRenderer

import numpy as np
import hydra

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    env=MiniGrid('human',cfg.map_8x8,cfg.render.fps,cfg.render.win_size)
    pr=PolicyRenderer(env.game.model,3,env.renderer)
    nS,nA=pr._model.nS,pr._model.nA
    env.game.add_plugin(pr)
    his=[]

    Qs=np.zeros((nS,nA))
    counts=np.zeros((nS,nA))
    pi=pr._model.ext_data['pi']
    
       
    s,_=env.reset(seed=123)
    for _ in range(int((nA*nS)**1.5)):
        #action=greedy_select(counts[s],env.np_random,True)
        action=categorical_sample(pi[s],env.np_random)
        s1,r,t,*_=env.step(action)
        counts[s,action]+=1
        his.append(Transition(s,action,s1,r,t))
        s=s1

    his.reverse()
    
    for t0 in his:
        t:Transition=t0
        V=Qs[t.s2].max()
        Qs[t.s1,t.action]=t.reward+cfg.algorithm.gamma*V
    env.reset()
    pi=pr._model.ext_data['pi']
    policy=Policy(env)
    pi*=0
    for s in range(nS):
        best_action=np.argmax(Qs[s])
        pi[s,best_action]=1


    while policy.running:
        action=policy.decition(s)
        s1,r,t,*_=env.step(action)
        if t :
            break


    env.close()


if __name__ == "__main__":
    main()