
from rlbase import MiniGrid
def main():
    env=MiniGrid('human','2x2',False)
    env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        env.step(action)

    env.close()

if __name__ == "__main__":
    main()
    