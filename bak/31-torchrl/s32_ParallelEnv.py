from torchrl.envs.libs.gym import GymEnv
from matplotlib import pyplot as plt

from torchrl.envs import ParallelEnv

def env_make():
    return GymEnv("Pendulum-v1")

if __name__ == '__main__':
    
    parallel_env = ParallelEnv(3, env_make)
    parallel_env.set_seed(10)
    print(parallel_env.reset())
    data=parallel_env.rollout(max_steps=20)
    print(data['observation'].shape)
    parallel_env.close()
    del parallel_env