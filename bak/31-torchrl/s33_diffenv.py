from torchrl.envs import ParallelEnv
from torchrl.envs.transforms import TransformedEnv, VecNorm
from torchrl.envs.libs.gym import GymEnv
from matplotlib import pyplot as plt
from torchrl.envs import Compose, GrayScale, Resize, ToTensorImage
def env_make(env_name):
    env = TransformedEnv(
        GymEnv(env_name, from_pixels=True, pixels_only=True),
        Compose(ToTensorImage(), Resize(64, 64)),
    )
    return env

if __name__ == '__main__':
    parallel_env = ParallelEnv(
        2,
        [env_make, env_make],
        [{"env_name": "ALE/AirRaid-v5"}, {"env_name": "ALE/Pong-v5"}],
    )
    tensordict = parallel_env.reset()

    plt.figure()
    plt.subplot(121)
    plt.imshow(tensordict[0].get("pixels").permute(1, 2, 0).numpy())
    plt.subplot(122)
    plt.imshow(tensordict[1].get("pixels").permute(1, 2, 0).numpy())
    parallel_env.close()
    del parallel_env
    plt.show()



    env = TransformedEnv(GymEnv("Pendulum-v1"), VecNorm())
    tensordict = env.rollout(max_steps=100)

    print("mean: :", tensordict.get("observation").mean(0))  # Approx 0
    print("std: :", tensordict.get("observation").std(0))  # Approx 1

