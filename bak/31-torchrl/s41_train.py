from tensordict.nn import TensorDictModule,TensorDictSequential
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP
from torchrl.collectors import SyncDataCollector,RandomPolicy
from torchrl.modules.tensordict_module import AdditiveGaussianWrapper
from torchrl.data import TensorDictReplayBuffer,LazyTensorStorage
from torchrl.objectives import DDPGLoss
from torchrl.trainers import Trainer
import torch

def main():
    env = GymEnv ('Pendulum-v1')
    print(env.reset())
    # Model : Actor and value
    mlp_actor = MLP (
        num_cells =64 ,  depth =3 ,
        in_features =3 , out_features =1
    )
    actor = TensorDictModule (
        mlp_actor ,
        in_keys =['observation'],
        out_keys =['action']
    )
    mlp_value = MLP (
        num_cells =64 , depth =2 ,
        in_features =4 ,out_features =1
    )
    critic = TensorDictSequential (
        actor ,TensorDictModule (
            mlp_value ,
            in_keys = [ 'observation','action'],
            out_keys = ['state_action_value']
        )
    )
    # Data Collector
    collector = SyncDataCollector (
        env ,actor,
        #RandomPolicy(env.action_spec),
        #AdditiveGaussianWrapper(actor) ,
        frames_per_batch =100 ,
        total_frames = 10000 ,
    )
    # Replay Buffer
    buffer = TensorDictReplayBuffer (
        storage = LazyTensorStorage ( max_size =10000) 
    )
    # Loss Module
    loss_fn = DDPGLoss(actor , critic )
    optim = torch.optim.Adam ( loss_fn.parameters() , lr=2e-4 )
    # Trainer
    trainer=Trainer(
        collector = collector,
        total_frames = 10000 ,
        frame_skip =1 ,
        optim_steps_per_batch =1 ,
        loss_module = loss_fn ,
        optimizer =optim
    )
    trainer.train()
    # for i, data in enumerate(collector):
    #     buffer.extend(data)
    #     sample=buffer.sample(50)
    #     loss=loss_fn(sample)
    #     loss=loss['loss_actor']+loss['loss_value']
    #     loss.backward()
    #     optim.step()
    #     optim.zero_grad()
    #     print(i)
if __name__ == '__main__':
    main()