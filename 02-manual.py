from rlbase import MiniGrid
from rlbase.policies.manual import ManualPolicy
from rlbase.envs.event_center import EventCenter

is_running=True
def on_quit():
    global is_running
    is_running=False

def main():
    hub=EventCenter()
    env=MiniGrid('human','4x4')
    policy=ManualPolicy(env,hub)
    hub.set_handler('APP_QUIT',on_quit)

    
    s,_=env.reset(seed=42)
    
    while is_running:
        action=policy.decition(s)
        s,*_=env.step(action)

    env.close()

if __name__ == "__main__":
    main()

    