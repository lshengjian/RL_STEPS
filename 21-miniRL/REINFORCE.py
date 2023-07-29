import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils import load_model,save_model,get_file_name
#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
print_interval = 200

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []
def demo():
    file_name=get_file_name(__file__).lower()
    env = gym.make('CartPole-v1',render_mode='human')
    pi,steps,best=load_model(file_name,Policy())
    observation, info = env.reset()
    for _ in range(1000):
        action = pi(torch.Tensor([observation]))[0][0].item() # a,pro
        
        observation, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
def main():
    file_name=get_file_name(__file__).lower()
    env = gym.make('CartPole-v1')
    pi,steps,best=load_model(file_name,Policy())
    score = 0.0
    
    for n_epi in range(steps+1,10000):
        s, _ = env.reset()
        done = False
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, truncated, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            ms=score/print_interval
            print(f"# of episode :{n_epi}, avg score : {ms:.1f}")
            if score>best:
                save_model(file_name,pi,n_epi,ms)
                best=score
                
            score = 0.0
    env.close()
    
if __name__ == '__main__':
    main()
    demo()
