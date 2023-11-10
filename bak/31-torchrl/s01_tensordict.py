from tensordict import TensorDict
import torch
import torch.nn as nn
from tensordict.nn import TensorDictSequential as Seq ,TensorDictModule as Mod
data = TensorDict({
    "observation": torch.ones (3 , 4 ) , # a tensor at the root level
    "next": {"observation": torch.ones (3 , 4 )}, # a nested TensorDict
}, batch_size =[3])
print(data)
print(data[0])
print(data ["next", "observation"])

class MLP(nn.Module ):
    def __init__ ( self ):
        super().__init__ ()
        self.layer1 = nn.Linear(3 , 4 )
        self.layer2 = nn.Linear (4 ,2 )
        self.activation = torch.relu

    def forward( self , x ):
        y = self.activation ( self.linear1 ( x ) )
        return self.linear2 ( y )

data = TensorDict({
    "input": torch.ones (9 , 3 ) 
}, batch_size =[9])
M = Seq (
    Mod( nn.Linear (3 , 8 ) , in_keys =['input'], out_keys =['hidden1']) ,
    Mod ( torch.relu , in_keys =['hidden1'], out_keys =['hidden2']) ,
    Mod ( nn.Linear (8 , 2 ) , in_keys =['hidden2'], out_keys =['output']) ,
)
print(M)
sub_module = M.select_subsequence ( in_keys =["hidden2"])
print(sub_module)
y=M(data)
print(y)




# module = Mod(
#     lambda x : x+1,x-2,
#     in_keys =["x"], out_keys =["y", "z"])
# y , z = module (x = torch . randn ( 3) )
# print(y,z)
# env_make = lambda: GymEnv("Pendulum-v1", from_pixels=True)
# env_parallel = ParallelEnv(4, env_make)  # creates 4 envs in parallel
# tensordict = env_parallel.rollout(max_steps=20, policy=None)  # random rollout (no policy given)
# assert tensordict.shape == [4, 20]  # 4 envs, 20 steps rollout
# env_parallel.action_spec.is_in(tensordict["action"])  # spec check returns True