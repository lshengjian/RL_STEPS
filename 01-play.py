
from rlbase import MiniGrid
from rlbase.policies.random import RandomPolicy
def main():
    env=MiniGrid('human','4x4',False)
    state,_=env.reset(seed=42)
    policy=RandomPolicy(env)
    for _ in range(100):
        action=policy.decition(state)
        state,*_=env.step(action)

    env.close()

if __name__ == "__main__":
    main()
