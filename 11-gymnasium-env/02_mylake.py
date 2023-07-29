import gymnasium as gym

from myenv.myenv  import MyLakeEnv
def main():
   env = MyLakeEnv(render_mode="human",map_name="4x4")
   observation, info = env.reset(seed=42)
   data=[0]*env.world.nS
   for _ in range(1000):
      action = env.action_space.sample(info['action_mask'])  # this is where you would insert your policy
      observation, reward, terminated, truncated, info = env.step(action)
      data[observation]+=1
      # if  reward<0:
      #    print(reward)
      if terminated or truncated:
         observation, info = env.reset()
   env.close()
   print(data)

if __name__ == "__main__":
   main()