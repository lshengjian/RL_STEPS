from torchrl.data import PrioritizedReplayBuffer, ReplayBuffer
from tensordict import TensorDict
import torch
buffer = ReplayBuffer()
print("length before adding elements:", len(buffer))
buffer.extend(range(2000))
print("length after adding elements:", len(buffer))

from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ListStorage
size = 10_000
buffer_list = ReplayBuffer(storage=ListStorage(size), collate_fn=lambda x: x)
buffer_list.extend(["a", 0, "b"])
print(buffer_list.sample(3))
buffer_lazytensor = ReplayBuffer(storage=LazyTensorStorage(size))
data = TensorDict(
    {
        'a': torch.arange(12).view(3, 4),
        ('b','c'): torch.arange(15).view(3, 5)
    },
    batch_size=[3],
)
print(data)

buffer_lazytensor.extend(data)
print(f"The buffer has {len(buffer_lazytensor)} elements")

sample = buffer_lazytensor.sample(5)
print("samples", sample["a"], sample["b", "c"])

buffer_lazymemmap = ReplayBuffer(
    storage=LazyMemmapStorage(size, scratch_dir="./data/")
)
buffer_lazymemmap.extend(data)
print(f"The buffer has {len(buffer_lazymemmap)} elements")
print("the 'a' tensor is stored in", buffer_lazymemmap._storage._storage["a"].filename)
print(
    "the ('b', 'c') tensor is stored in",
    buffer_lazymemmap._storage._storage["b", "c"].filename,
)

from torchrl.data import TensorDictReplayBuffer

buffer_lazymemmap = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(size, scratch_dir="./data/"), batch_size=12
)
buffer_lazymemmap.extend(data)
print(f"The buffer has {len(buffer_lazymemmap)} elements")
sample = buffer_lazymemmap.sample()
print(sample["index"])
for e in  sample['a']:
    print(e.numpy())