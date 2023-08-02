from mygrid.env  import MiniGrid
from mygrid.config import ACT_NAMES
from mygrid.utils import categorical_sample
from random import random
from gymnasium.utils import seeding
import numpy as np
import time
GAMMA=0.9
EPS=1e-4
def update_value(grid:MiniGrid):
   flag=True
   while flag:
      flag=False
      for s in range(grid.nS):
         cur=grid.get_value(s)
         r,c=grid.world.state2idx(s)
         data=np.zeros(grid.nA,dtype=float)
         for a in range(grid.nA):
            s_,r=grid.world.try_move(r,c,a)
            data[a]=r+GAMMA*grid.get_value(s_)
         now=data.max()
         changed=now-cur
         if changed>EPS:
            flag=True
            grid.set_value(s,now)
            grid.render()
   print('state value updated!')
def update_policy(grid:MiniGrid):
   for s in range(grid.nS):
      pi=grid.P[s]
      r,c=grid.world.state2idx(s)
      data=np.zeros(grid.nA,dtype=float)
      for a in range(grid.nA):
         s_,_=grid.world.try_move(r,c,a)
         data[a]=grid.get_value(s_)
         #print(r,c,a,data[a])
      data/=data.sum()
      pi*=0.0
      idx=categorical_sample(data,grid.np_random)
      pi[idx]=1.0
   #print('policy updated!')


def main():
   env0= MiniGrid(render_mode="human",map_name="4x4",is_terminate_reach_goal=False)
   env0.reset(seed=42)
   update_value(env0)
   env= MiniGrid(render_mode="human",map_name="4x4",is_terminate_reach_goal=True)
   s,_=env.reset(seed=44)
   env.V=env0.V

   #env.renderer.FPS=20
   for _ in range(1000):
      update_policy(env)
      pi=env.P[s]
      a=categorical_sample(pi,env.np_random)
      s,  r, terminated, truncated, _  = env.step(a)
      if terminated or truncated:
         time.sleep(2)
         s,_=env.reset()
         #print('reset',s)
         #break


   env.close()


if __name__ == "__main__":
   main()
   