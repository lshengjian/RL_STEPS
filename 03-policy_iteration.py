from rlbase import PolicyGrid
from rlbase.policies.manual import RandomPolicy

def main():
    env=PolicyGrid('human','4x4')
    policy=RandomPolicy(env)
   
    s,_=env.reset(seed=42)
    
    for _ in range(1000):
        action=policy.decition(s)
        s,*_=env.step(action)

    env.close()

if __name__ == "__main__":
    main()