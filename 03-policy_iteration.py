from rlbase import MiniGrid
from rlbase.policies.model.policy_iteration import IterationPolicy
def main():
    env=MiniGrid('human','4x4',True)
    policy=IterationPolicy(env)
    state,_=env.reset(seed=42)
    
    for _ in range(400):
        action=policy.decition(state)
        state, r, terminated,*_=env.step(action)
        if terminated:
            state,_=env.reset()


    env.close()

if __name__ == "__main__":
    main()
