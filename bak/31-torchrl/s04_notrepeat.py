from tensordict import TensorDict
import torch
from torchrl.data import LazyMemmapStorage,ListStorage
from torchrl.data import TensorDictReplayBuffer,ReplayBuffer
import torch
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

size=128
buffer_lazymemmap = ReplayBuffer(
    storage=LazyMemmapStorage(size), batch_size=32, sampler=SamplerWithoutReplacement()
)

data = TensorDict(
    {
        "a": torch.arange(512).view(size, 4),
        ("b", "c"): torch.arange(1024).view(size, 8),
    },
    batch_size=[size],
)

buffer_lazymemmap.extend(data)
for _i,d in enumerate(buffer_lazymemmap):
    print(d['b']['c'].numpy())
print(f"A total of {_i+1} batches have been collected")
print("sampling 3 elements:", buffer_lazymemmap.sample(3))
print("sampling 5 elements:", buffer_lazymemmap.sample(5))