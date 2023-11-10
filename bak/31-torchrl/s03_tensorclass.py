from tensordict import tensorclass
import torch
from torchrl.data import LazyMemmapStorage,ListStorage
from torchrl.data import TensorDictReplayBuffer,ReplayBuffer
@tensorclass
class MyData:
    images: torch.Tensor
    labels: torch.Tensor
size=1000

data = MyData(
    images=torch.randint(
        255,
        (size, 64, 64, 3),
    ),
    labels=torch.randint(100, (size,)),
    batch_size=[size],
)

buffer_lazymemmap = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(size, scratch_dir="./memmap/"), batch_size=12
)
buffer_lazymemmap.extend(data)
print(f"The buffer has {len(buffer_lazymemmap)} elements")
sample = buffer_lazymemmap.sample()
print("sample:", sample.labels.numpy())

from torchrl.data.replay_buffers.samplers import PrioritizedSampler

size = 1000

rb = ReplayBuffer(
    storage=ListStorage(size),
    sampler=PrioritizedSampler(max_capacity=size, alpha=0.8, beta=1.1),
    collate_fn=lambda x: x,
)

indices = rb.extend([1, "foo", None])

rb.update_priority(index=indices, priority=torch.tensor([0, 1_000, 0.1]))

sample, info = rb.sample(10, return_info=True)
print(sample)
print(info)

buffer_lazymemmap = ReplayBuffer(
    storage=LazyMemmapStorage(size), batch_size=128, prefetch=10
)  # creates a queue of 10 elements to be prefetched in the background
buffer_lazymemmap.extend(data)
print(buffer_lazymemmap.sample())