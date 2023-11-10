import torch
from torch.multiprocessing import Pipe
from torch.distributions.categorical import Categorical
#import gym
import gymnasium as gym
import numpy as np
import os

from envs import AtariEnvironment
from arguments import get_args
from model import CnnActorCriticNetwork

def get_action(model, device, state):
    state = torch.Tensor(state).to(device)
    action_probs, value_ext, value_int = model(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    return action.data.cpu().numpy().squeeze()

def main():
    args = get_args()
    device = torch.device('cpu')#'cuda' if args.cuda else 'cpu'
    input_size = (210, 160, 3)#env.observation_space.shape  # 4
    output_size = 18 # env.action_space.n  # 2
    #print(input_size,output_size)

    # if 'Breakout' in args.env_name:
    #     output_size -= 1


    
    is_render = True
    model_path = os.path.join(args.save_dir, args.env_name + '.model')
    if not os.path.exists(model_path):
        print("Model file not found")
        return
    num_worker = 1
    sticky_action = False
        
    model = CnnActorCriticNetwork(input_size, output_size, args.use_noisy_net)
    model = model.to(device)

    if args.cuda:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    parent_conn, child_conn = Pipe()
    work = AtariEnvironment(
      	args.env_name,
        is_render, 
       	0, 
       	child_conn, 
       	sticky_action=sticky_action, 
       	p=args.sticky_action_prob,
       	max_episode_steps=args.max_episode_steps)
    work.start()

    #states = np.zeros([num_worker, 4, 84, 84])
    states = torch.zeros(num_worker, 4, 84, 84)
    size=list(states.shape)

    while True:
        actions = get_action(model, device, states/255.)

        parent_conn.send(actions)

        next_states = []
        next_state, reward, done, real_done, log_reward = parent_conn.recv()
        if done or real_done:
            print(done,real_done)
        next_states.append(next_state)
        states = torch.from_numpy(np.stack(next_states)) 
        states = states.float() #states.type(torch.FloatTensor)
    
if __name__ == '__main__':
    main()
