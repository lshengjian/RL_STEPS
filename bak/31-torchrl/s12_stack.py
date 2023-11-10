from torchrl.collectors import RandomPolicy, SyncDataCollector
from torchrl.envs import Compose, GrayScale, Resize, ToTensorImage, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.data import TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage
from torchrl.envs import CatFrames, UnsqueezeTransform

env = TransformedEnv(
    GymEnv("CartPole-v1", from_pixels=True),
    Compose(
        ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
        Resize(in_keys=["pixels_trsf"], w=64, h=64),
        GrayScale(in_keys=["pixels_trsf"]),
        UnsqueezeTransform(-4, in_keys=["pixels_trsf"]),
        CatFrames(dim=-4, N=4, in_keys=["pixels_trsf"]),
    ),
)
collector = SyncDataCollector(
    env,
    RandomPolicy(env.action_spec),
    frames_per_batch=10,
    total_frames=1000,
)
for data in collector:
    print(data)
    break

t = Compose(
    ToTensorImage(
        in_keys=["pixels", ("next", "pixels")],
        out_keys=["pixels_trsf", ("next", "pixels_trsf")],
    ),
    Resize(in_keys=["pixels_trsf", ("next", "pixels_trsf")], w=64, h=64),
    GrayScale(in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
    CatFrames(dim=-4, N=4, in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
)
rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(1000), transform=t, batch_size=16)
data_exclude = data.exclude("pixels_trsf", ("next", "pixels_trsf"))
rb.add(data_exclude)