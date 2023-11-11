from mygrid.env  import MiniGrid
import gymnasium as gym
import numpy as np
import hydra
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters

gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,4)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@hydra.main(version_base="1.2", config_path=".", config_name="03-dqn")
def main(cfg: "DictConfig"):  # noqa: F821
    env=MiniGrid(render_mode="human",
                map_name=cfg.env.map_name, 
                fps=cfg.env.fps,
                is_terminate_reach_goal=True)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = cfg.optim.print_interval
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=cfg.optim.learning_rate)

    for n_epi in range(10000):
        
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        #print(n_epi,epsilon)
        s, _ = env.reset()
        s=env.world.get_vec()
        
        done = False

        for i in range(100):
            
            a = q.sample_action(torch.from_numpy(np.array(s)).float(), epsilon)      
            s_prime, r, done, truncated, info = env.step(a)
            s_prime=env.world.get_vec()
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
            for s in range(env.world.nS):
                sv=env.world.get_vec_state(s)
                out = q_target.forward(torch.from_numpy(np.array(sv)).float())
                env.set_value(s,out.max().item())

    env.close()

if __name__ == '__main__':
    main()





   