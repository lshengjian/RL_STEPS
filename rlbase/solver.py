from .envs import MyGrid
import torch
from torch.utils import data
import torch.nn as nn
import numpy as np
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class QNET(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(QNET, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


class Solver:
    def __init__(self, env: MyGrid):
        self.gama = 0.9
        self.env = env
        # self.pr=PolicyRenderer(env.game.model,3,env.renderer)
        # env.game.add_plugin(self.pr)
        self.action_space_size = env.model.nA
        self.state_space_size = env.model.nS
        #self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.zeros(shape=(self.state_space_size,))
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
    
    def get_data_iter(self, episode, batch_size=64, is_train=True):
        """构造一个PyTorch数据迭代器"""
        reward = []
        state_action = []
        next_state = []
        for i in range(len(episode)):
            reward.append(episode[i]['reward'])
            action = episode[i]['action']
            y, x = self.env.model.state2idx(episode[i]['state'])
            state_action.append((y, x, action))
            y, x = self.env.model.state2idx(episode[i]['next_state'])
            next_state.append((y, x))
        reward = torch.tensor(reward).reshape(-1, 1)
        state_action = torch.tensor(state_action)
        next_state = torch.tensor(next_state)
        data_arrays = (state_action, reward, next_state)
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=False)
        
    def obtain_episode(self, policy, start_state, start_action, length):
        f"""

        :param policy: 由指定策略产生episode
        :param start_state: 起始state
        :param start_action: 起始action
        :param length: episode 长度
        :return: 一个 state,action,reward,next_state,next_action 序列
        """
        #self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.model.state #self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode   
     
    def q_learning_off_policy(self, alpha=0.01, epsilon=0.1, num_episodes=10000, episode_length=2000):
        start_state = self.env.model.state #self.env.pos2state(self.env.agent_location)
        start_action = np.random.choice(np.arange(self.action_space_size),
                                        p=self.mean_policy[start_state])
        episode = self.obtain_episode(self.mean_policy.copy(), start_state=start_state, start_action=start_action,
                                      length=episode_length)
        for step in range(len(episode) - 1):
            reward = episode[step]['reward']
            state = episode[step]['state']
            action = episode[step]['action']
            next_state = episode[step + 1]['state']
            next_qvalue_star = self.qvalue[next_state].max()
            target = reward + self.gama * next_qvalue_star
            error = self.qvalue[state, action] - target
            self.qvalue[state, action] = self.qvalue[state, action] - alpha * error
            action_star = self.qvalue[state].argmax()
            self.policy[state] = np.zeros(self.action_space_size)
            self.policy[state][action_star] = 1

    def dqn(self, learning_rate=0.0015, episode_length=5000, epochs=6000, batch_size=100, update_step=10):
        q_net = QNET().to(device)
        # policy = self.policy.copy()
        # state_value = self.state_value.copy()
        q_target_net = QNET().to(device)
        q_target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.SGD(q_net.parameters(),
                                    lr=learning_rate)
        episode = self.obtain_episode(self.mean_policy, 0, 0, length=episode_length)
        date_iter = self.get_data_iter(episode, batch_size)
        loss = torch.nn.MSELoss()
        approximation_q_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        i = 0
        #loss_list=[]
        for epoch in range(epochs):
            q_target_net.eval()
            q_net.train()
            for state_action, reward, next_state in date_iter:
                i += 1
                q_value = q_net(state_action.to(device))
                q_value_target = torch.empty((batch_size, 0)).to(device)  # 定义空的张量
                for action in range(self.action_space_size):
                    s_a = torch.cat((next_state, torch.full((batch_size, 1), action)), dim=1)
                    q_value_target = torch.cat((q_value_target, q_target_net(s_a.to(device))), dim=1)
                q_star = torch.max(q_value_target, dim=1, keepdim=True)[0]
                y_target_value = reward.to(device) + self.gama * q_star
                l = loss(q_value, y_target_value)
                optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0S
                l.backward()  # 反向传播更新参数
                optimizer.step()
                if i % update_step == 0 and i != 0:
                    q_target_net.load_state_dict(
                        q_net.state_dict())  # 更新目标网络
                    # policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            #loss_list.append(float(l))
            print(f"loss:{l},epoch:{epoch}")
            # self.policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            # self.state_value = np.zeros(shape=self.state_space_size)

            # q_net.eval()
            # for s in range(self.state_space_size):
            #     y, x = self.env.model.state2idx(s)
            #     for a in range(self.action_space_size):
            #         approximation_q_value[s, a] = float(q_net(torch.tensor((y, x, a)).reshape(-1, 3).to(device)))
            #     q_star_index = approximation_q_value[s].argmax()
            #     self.policy[s, q_star_index] = 1
            #     self.state_value[s] = approximation_q_value[s, q_star_index]
            #self.pr._model.ext_data['pi']=self.policy
        return q_net

