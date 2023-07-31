from mygrid.env  import MiniGrid
from mygrid.config import ACT_NAMES
from random import random
def main():
   env = MiniGrid(render_mode="human",map_name="4x4")
   observation, info = env.reset(seed=42)
   data=[0]*env.world.nS
   for _ in range(1000):
      action = env.action_space.sample(info['action_mask'])  
      observation1, reward, terminated, truncated, info = env.step(action)
      data[observation]+=1
      if abs(reward)>0:
         idxs=env.world.state2idx(observation)
         pos=idxs[0]+1,idxs[1]+1
         #print(pos,ACT_NAMES[action],reward)
      if terminated or truncated:
         observation, info = env.reset()
      else:
         observation=observation1
         env.update_value(observation,random()*10)
   env.close()
   print(data)

if __name__ == "__main__":
   main()