from mygrid.env  import MiniGrid
from mygrid.config import ACT_NAMES
from mygrid.utils import greedy_select
from model import QNET
from random import random
import gymnasium as gym
import numpy as np
from torch.utils import data
import hydra
import torch

def obtain_episode(env,  length):
    f"""

    :param policy: 由指定策略产生episode
    :param start_state: 起始state
    :param start_action: 起始action
    :param length: episode 长度
    :return: 一个 state,action,reward,next_state,next_action 序列
    """
    #env.agent_location = env.state2pos(start_state)
    policy=env.PI
    episode = []
    
    next_state,_ = env.reset()
    next_action = np.random.choice(np.arange(len(policy[next_state])),
                                        p=policy[next_state])
    while length > 0:
        length -= 1
        state = next_state
        action = next_action
        next_state, reward, done, _, _ = env.step(action)
        next_action = np.random.choice(np.arange(len(policy[next_state])),
                                        p=policy[next_state])
        episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                        "next_action": next_action})
    return episode

def get_data_iter(env:MiniGrid,episode, batch_size=64, is_train=True):
    """构造一个PyTorch数据迭代器"""
    reward = []
    state_action = []
    next_state = []
    for i in range(len(episode)):
        reward.append(episode[i]['reward'])
        action = episode[i]['action']
        y, x = env.world.state2idx(episode[i]['state'])
        state_action.append((y, x, action))
        y, x = env.world.state2idx(episode[i]['next_state'])
        next_state.append((y, x))
    reward = torch.tensor(reward).reshape(-1, 1)
    state_action = torch.tensor(state_action)
    next_state = torch.tensor(next_state)
    data_arrays = (state_action, reward, next_state)
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=False)


@hydra.main(version_base="1.2", config_path=".", config_name="02-config")
def main(cfg: "DictConfig"):  # noqa: F821
   env=MiniGrid(render_mode="human",
                map_name=cfg.env.map_name, 
                fps=cfg.env.fps,
                is_terminate_reach_goal=True)
   s,_=env.reset()
   q_net = QNET()
   q_target_net = QNET()
   q_target_net.load_state_dict(q_net.state_dict())
   optimizer = torch.optim.SGD(q_net.parameters(),
                                    lr=cfg.optim.learning_rate)
   episode = obtain_episode(env, length=cfg.simulate.episode_length)
   batch_size=cfg.optim.batch_size
   date_iter = get_data_iter(env,episode,batch_size)
   loss = torch.nn.MSELoss()
   approximation_q_value = np.zeros(shape=(env.nS, env.nA))
   i = 0
   for epoch in range(cfg.simulate.epochs):
        for state_action, reward, next_state in date_iter:
            i += 1
            q_value = q_net(state_action)
            q_value_target = torch.empty((batch_size, 0))  # 定义空的张量
            for action in range(env.nA):
                s_a = torch.cat((next_state, torch.full((batch_size, 1), action)), dim=1)
                q_value_target = torch.cat((q_value_target, q_target_net(s_a)), dim=1)
            q_star = torch.max(q_value_target, dim=1, keepdim=True)[0]
            y_target_value = reward + cfg.optim.gamma * q_star
            l = loss(q_value, y_target_value)
            optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0S
            l.backward()  # 反向传播更新参数
            optimizer.step()
            if i % 20 == 0 and i != 0:
                q_target_net.load_state_dict(
                    q_net.state_dict())  # 更新目标网络
                    # policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
                print("loss:{},epoch:{}".format(l.item(), epoch))

            for s in range(env.nS):
                y, x =  env.world.state2idx(s)
                for a in range(env.nA):
                    approximation_q_value[s, a] = float(q_net(torch.tensor((y, x, a)).reshape(-1, 3)))
                q_star_index = approximation_q_value[s].argmax()
                env.PI[s, :] = 0
                env.PI[s, q_star_index] = 1
                env.V[s] = approximation_q_value[s, q_star_index]
            #rmse_list.append(np.sqrt(np.mean((state_value - self.state_value) ** 2)))




   env.close()



if __name__ == "__main__":
   main()
   