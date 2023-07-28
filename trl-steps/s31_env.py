from torchrl.envs.libs.gym import GymEnv
from matplotlib import pyplot as plt

env = GymEnv("Pendulum-v1", frame_skip=4)
print(env.reset())

env = GymEnv("Pendulum-v1", from_pixels=True)
tensordict = env.reset()
env.close()
plt.imshow(tensordict.get("pixels").numpy())
plt.show()