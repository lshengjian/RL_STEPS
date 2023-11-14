from rlbase import MiniGrid
from rlbase.policies.manual import ManualPolicy
import esper

is_running=True
def on_quit():
    global is_running
    is_running=False

def main():
    env=MiniGrid('human','4x4',True)
    policy=ManualPolicy(env)
    esper.set_handler('APP_QUIT',on_quit)

    
    s,_=env.reset(seed=42)
    while is_running:
        action=policy.decition(s)
        s,*_=env.step(action)

    env.close()

if __name__ == "__main__":
    main()

    