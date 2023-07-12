from collections import defaultdict
import torch

from torchrl.envs.libs.gym import GymEnv
from tensordict.nn import TensorDictSequential,TensorDictModule
from torchrl.modules import MLP,AdditiveGaussianWrapper

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer,TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.trainers import Trainer

from torchrl.objectives import ClipPPOLoss,DDPGLoss
# from torchrl.objectives.value import GAE
from tqdm import tqdm

env = GymEnv("Pendulum-v1")
# Model : Actor and value
mlp_actor = MLP (
    num_cells =64 ,
    depth =3 ,
    in_features =3 ,
    out_features =1
)
actor = TensorDictModule (
    mlp_actor ,
    in_keys =['observation'],
    out_keys =['action']
)
mlp_value = MLP (
    num_cells =64 , 
    depth =2 ,
    in_features =4 ,
    out_features =1
)
critic = TensorDictSequential (
    actor ,
    TensorDictModule (
        mlp_value ,
        in_keys = ['observation','action'],
        out_keys =['state_action_value']
    )
)
# Data Collector
collector = SyncDataCollector (
    env ,
    AdditiveGaussianWrapper (actor) ,
    frames_per_batch =1000 ,
    total_frames = 1000000
)
 # Replay Buffer
buffer = TensorDictReplayBuffer (
    storage = LazyTensorStorage (max_size =100000)
)
 # Loss Module
loss_fn = DDPGLoss (actor , critic  )
optim = torch . optim . Adam (loss_fn . parameters () ,lr=2e-4 ,)
# Trainer
trainer=Trainer (
    collector = collector,
    total_frames = 1000000 ,
    frame_skip =1 ,
    optim_steps_per_batch =1 ,
    loss_module = loss_fn ,
    optimizer =optim ,
)
trainer.train()