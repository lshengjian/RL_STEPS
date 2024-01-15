from rlbase.solver import Solver,QNET,device
from rlbase import MyGrid
from rlbase.policies.manual import ManualPolicy
import hydra,torch
import numpy as np
from rlbase.plugins.renderer_policy import PolicyRenderer


@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    # 
    # env=MyGrid(None,cfg.map_5x5,cfg.render.fps,cfg.render.win_size) 
    # solver=Solver(env)
    # dqn=solver.dqn()
    # env.close()
    # torch.save(dqn.state_dict(),'dqn.pt')
    env=MyGrid('human',cfg.map_5x5,cfg.render.fps,cfg.render.win_size)
    pr=PolicyRenderer(env.game.model,3,env.renderer)
    env.game.add_plugin(pr)
    dqn=QNET()
    dqn.load_state_dict(torch.load('dqn.pt',map_location='cpu'))
    mgr=ManualPolicy(env)
    obs,_=env.reset()

    while mgr.running:
        s=env.model.state
        action=mgr.decition(s)
        x,y=obs
        qs=np.zeros(env.model.nA)
        for a in range(env.model.nA):
            qs[a] = float(dqn(torch.tensor((y, x, a)).reshape(-1, 3)))
        action = qs.argmax()
        #print(action)
        pi=env.model.ext_data['pi'][s]

        pi*=0
        pi[action]=1
        obs, r,done,*_ = env.step(action)
        if done:
            obs,_=env.reset()
        


    env.close()
    

if __name__ == '__main__':
    main()
