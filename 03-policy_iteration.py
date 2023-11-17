from rlbase import PolicyGrid
from rlbase.policies.manual import ManualPolicy

def main():
    env=PolicyGrid('human','4x4')
    policy=ManualPolicy(env)
   
    s,_=env.reset(seed=42)
    
    while policy.running:
        action=policy.decition(s)
        s,*_=env.step(action)

    env.close()

if __name__ == "__main__":
    main()