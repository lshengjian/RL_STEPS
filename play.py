import gymnasium as gym
from rlbase import MiniGrid
def main():
    env=MiniGrid('human','2x2',False,False,False)
    #env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)



    env.close()

if __name__ == "__main__":
    main()
    