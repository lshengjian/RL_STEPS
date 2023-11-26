import numpy as np
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rlbase import MiniGrid
import hydra
#Hyperparameters
learning_rate = 0.001
gamma         = 0.9
buffer_limit  = 50000
batch_size    = 32
nActions=5
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
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
    def sample_action(self, obs, epsilon=0):
        self.eval()
        out = self.forward(obs)
        coin = random.random()
        rt=0
        if coin < epsilon:
            rt= random.randint(0,nActions-1)
        else : 
            rt= out.argmax().item()
        return rt
            
def train(Q, Q_target, memory, optimizer):
    Q.train()
    for _ in range(20):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = Q(s)
        q_a = q_out.gather(-1,a)
        max_q_prime = Q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    env=MiniGrid(None,cfg.map_5x5,cfg.render.fps,cfg.render.win_size) #
    rows=env.model.nrow
    cols=env.model.ncol
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False

        for i in range(1000):
            r,c=env.model.state2idx(s)
            y,x=r/rows,c/cols
            s1=np.array([x,y])
            a = q.sample_action(torch.from_numpy(s1).float(), epsilon)      
            s_prime, r, done, truncated, info = env.step(a)
            r,c=env.model.state2idx(s_prime)
            y,x=r/rows,c/cols
            s2=np.array([x,y])
            done_mask = 0.0 if done else 1.0
            memory.put((s1,a,r/10.0,s2, done_mask))
            s = s_prime

            score += r
            if done:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print(f"n_episode :{n_epi}, score : {score/print_interval:.1f}, eps : {epsilon*100:.1f}%")
            score = 0.0
    env.close()
    torch.save(q.state_dict(),'dqn.pt')
    q.load_state_dict(torch.load('dqn.pt'))
    env=MiniGrid('human',cfg.map_5x5,cfg.render.fps,cfg.render.win_size)
    s, _ = env.reset()
    for i in range(1000):
        r,c=env.model.state2idx(s)
        y,x=r/rows,c/cols
        s1=np.array([x,y])
        a = q.sample_action(torch.from_numpy(s1).float())      
        s_prime, rew, done, truncated, info = env.step(a)
        print(r,c,a,rew)
        s = s_prime

    env.close()
    

if __name__ == '__main__':
    main()
