import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils import load_model,save_model,get_file_name
#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
num_envs      = 8

def make_env():
   return gym.make("CartPole-v1")
class PPO(nn.Module):
    def __init__(self,in_size=4,out_size=2):
        super().__init__()
        self.data = []
        
        self.fc1   = nn.Linear(in_size,256)
        self.fc_pi = nn.Linear(256,out_size)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            #done_mask = 0 if done else 1
            done_mask=1-done
            done_lst.append(done_mask)
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(np.concatenate(s_lst), dtype=torch.float),\
              torch.tensor(np.concatenate(a_lst), dtype=torch.int64), \
              torch.tensor(np.concatenate(r_lst)), \
              torch.tensor(np.concatenate(s_prime_lst), dtype=torch.float), \
              torch.tensor(np.concatenate(done_lst), dtype=torch.float), \
              torch.tensor(np.concatenate(prob_a_lst))
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()
            

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=-1)
            pi_a=pi[a]
            #pi_a = pi.gather(-1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
def demo():
    file_name=get_file_name(__file__).lower()
    env = gym.make('CartPole-v1',render_mode='human')
    model,steps,best=load_model(file_name,PPO())
    print(steps,best)
    observation, info = env.reset()
    s=0
    for _ in range(1000):
        prob = model.pi(torch.from_numpy(observation).float())
        action=prob.argmax().item()
        
        #action = model.pi(torch.Tensor([observation]),softmax_dim=1)[0].argmax() # a,pro
        observation, reward, terminated, truncated, info = env.step(int(action))
        s+=reward
        if terminated or truncated:
            print(s)
            s=0
            break

    env.close()        
def main():
    file_name=get_file_name(__file__).lower()
    env = gym.vector.AsyncVectorEnv([ make_env ]*num_envs)
    model,steps,best=load_model(file_name,PPO())
    print(steps,best)


    score = 0.0
    print_interval = 20

    for n_epi in range(steps+1,10000):
        s, _ = env.reset()
        end = False
        while not end:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(np.stack(s)).float())
                m = Categorical(prob)
                a = m.sample().detach().numpy()
                prob_a=prob.detach().numpy()[a]
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob_a, done))
                if truncated.any():
                    end=True
                    print('---truncated---')
                if done.any() :
                    end=True
                    break
                s = s_prime

                score += r[0]
                

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            ms=score/print_interval
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, ms))
            if score>best:
                save_model(file_name,model,n_epi,ms)
                best=score
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
    #demo()
    