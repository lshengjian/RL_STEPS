from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.data import TensorDictReplayBuffer,ReplayBuffer
from torchrl.data import ListStorage
import torch
from tensordict import TensorDict
size = 128

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
data = TensorDict(
    {
        "a": torch.arange(512).view(128, 4)
    },
    batch_size=[128],
)
rb = TensorDictReplayBuffer(
    storage=ListStorage(size),
    sampler=PrioritizedSampler(128, alpha=0.8, beta=1.1),
    priority_key="td_error",
    batch_size=128*8,
)
data["td_error"] = torch.arange(data.numel())

rb.extend(data)

sample = rb.sample()
print(sample['a'].numpy())
from matplotlib import pyplot as plt

plt.hist(sample["index"].numpy())
plt.show()