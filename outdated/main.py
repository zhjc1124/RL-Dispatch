# -------------------------------------------------
# Description:
# Reference:
# Author:   Wang Shengpeng
# encoding: utf-8
# Date:     2021/12/2

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import myenv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


''' 
    定义网络并实例化：
    net1作为同一时隙包裹的共用状态passengers和packages两通道的特征提取
    net2作为同一时隙包裹的私有状态和net1的特征输出拼接后的网络输出
'''
class Net1(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.input_layer = nn.Sequential(nn.Conv2d(2, 1, (4, 3), stride=(2, 2)),
                                nn.MaxPool2d(kernel_size=3, stride=2),
                                nn.ReLU())
        # self.output_layer = nn.Linear(28 * 28 , 118)
        
    def forward(self, x):
        x = self.input_layer(x)
        # x = self.output_layer(x)
        return x
net1 = Net1()

class Net2(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.input_layer = nn.Sequential(nn.Linear(787, 256),
                                nn.Dropout(0.5),
                                nn.ReLU())
        self.hidden_layer1 = nn.Sequential(nn.Linear(256, 256),
                                nn.Dropout(0.3),
                                nn.ReLU())
        self.output_layer = nn.Linear(256, 118)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.output_layer(x)  
        return x
net2 = Net2()

net1.to(device)
net2.to(device)

# env = myenv.Myenv()
# state = env.reset()

# 定义DQN智能体
class DQNAgent:
    def __init__(self, env):
        self.action_n = 118                                             # 来自环境的动作数，118站点
        self.gamma = 0.99                                               # 折损因子
        self.optimizer1 = optim.Adam(net1.parameters(), lr=0.001)       # net1的优化器
        self.optimizer2 = optim.Adam(net2.parameters(), lr=0.001)       # net2的优化器
        self.loss = nn.MSELoss()                                        # 均方误差
        
        
    '''
        CustomNet:
            input: env.state
                type: dict
            output: q_state     (for a time shot)
                type: tensor 
                size: [dispatchs_length, 118]
    '''    
    def CustomNet(self, env_state):
        # 拼接tensor        
        a = torch.cat((env_state['passengers'][0].unsqueeze(0), env_state['passengers'][1]), 0)     # (119, 118)
        b = torch.cat((env_state['packages'][0].unsqueeze(0), env_state['packages'][1]), 0)         # (119, 118)
        c = torch.cat((a.unsqueeze(0), b.unsqueeze(0)), 0)                                          # (2, 119, 118)
        c = c.to(device)
        d = net1(c.unsqueeze(0).to(device))                                                         # (1, 28, 28)
        # e = d.view(-1)
        e = torch.flatten(d, start_dim=1)                                                           # (1, 28 * 28)
        state_dis = torch.as_tensor(env_state['dispatchs'], dtype=torch.float32)                    # (dispatchs_length, 3)
        if torch.numel(state_dis):
            dispatchs_length = state_dis.size(0)
            dis_tensor = e.repeat(dispatchs_length, 1)                                              # (dispatchs_length, 28 * 28)
            state_tensor_input = torch.cat((state_dis.to(device), dis_tensor.to(device)), 1)        # (dispatchs_length, 3 + 28 * 28)
            state_tensor_input = state_tensor_input.to(device)
            q_state_evaluate = net2(state_tensor_input)                                             # (dispatchs_length, 118)
        else:
            # 没有dispatches
            state_tensor_input = torch.cat((torch.as_tensor([[-1., -1., -1.]]).to(device), e), 1)   # (1, 3 + 28 * 28)
            state_tensor_input = state_tensor_input.to(device)
            q_state_evaluate = net2(state_tensor_input)                                             # (1, 118)
        return q_state_evaluate
    

    
    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []        # record state, action, next_state, reward, done for update the parameter of q net

    # agent决策
    def step(self, observation, reward, done):
        if self.mode == 'train' and np.random.rand() < 0.001:
            # epsilon-greedy policy in train mode
            dispatchs_len = len(observation['dispatchs'])
            action = np.random.randint(self.action_n, size = (1, dispatchs_len))
            action = torch.as_tensor(action, dtype = torch.int64)                                    # 维度[1, dispatchs_len]
            # assert(dispatchs_len == action.size(1))
            
        else:
 
            state_tensor = observation                              # get state
            q_tensor = self.CustomNet(state_tensor)                 # 得到预估q值
            '''
                应该有个动作过滤器(ETA过滤)，消除无意义的动作，加快收敛
            '''
            action_tensor = torch.argmax(q_tensor, 1)               # 选择最大的Q值对应的动作---此动作是针对dispatchs----维度[dispatchs_len]
            action = action_tensor.unsqueeze(0)                     # 选择最大的Q值对应的动作----维度[1, dispatchs_len]
            
        if self.mode == 'train':
                self.trajectory += [observation.copy(), reward, done, action]
                # 当存在next state,且next action和action不为空时进行学习
                if len(self.trajectory) >= 8 and torch.numel(self.trajectory[-5]) and torch.numel(self.trajectory[-1]):
                    self.learn()                                                    # 开始学习
        return action                                                               # 维度[1, dispatchs_len]

    def close(self):
        pass

    def learn(self):
        state, _, _, action, next_state, reward, done, action_next = self.trajectory[-8:]
        
        next_qs = self.CustomNet(next_state)                                                 # (dispatchs, 118)
        # print(action_next.size())
        '''
            因为难以说明每个包裹的状态和下一个状态的Q，因此我将未知维度的包裹簇的每个Q做了平均代表当前时隙的Q(state)
            另外删减了target Q和replayer，也即当前状态的Q估计和下一状态的Q估计都采用同一个网络(缺点是数据的相关性可能造成难以收敛)
        '''
        next_q = next_qs.gather(1, action_next.t().cuda()).mean()                                   # 用包裹簇的mean_q作为当前状态时隙的Q
        target_q = reward + self.gamma * next_q * (1. -done)                                 # 目标Q
        # state = state.to(device)
        predict_qs = self.CustomNet(state)                                                   # (dispatchs, 118)，
        predict_q = predict_qs.gather(1, action.t().cuda()).mean()                                  # 预测Q
        loss_tensor = self.loss(target_q, predict_q)                                         # 以目标Q与预测Q的均方差为损失
        self.optimizer1.zero_grad()                                                          # 
        self.optimizer2.zero_grad()
        loss_tensor.backward()
        self.optimizer1.step()
        self.optimizer2.step()


# agent = DQNAgent(env)



# 运行一个回合
def play_episode(env, agent, max_episode_steps=None, mode=None, render=False):
    observation, reward, done = env.reset(), 0., False                                      ## 来自环境的reset函数
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        if done:
            break
        if observation['dispatchs']:
            actions = agent.step(observation, reward, done)                                 # (1, dispatchs)
            actions_list = actions.squeeze(0).tolist()                                      # 转化为list
        else:
            actions_list = []
            
        observation = env.step(actions_list)
        print(observation['time'])
        reward = observation['reward']
        done = observation['done']
        episode_reward += reward
        elapsed_steps += 1
        
        if render:
            env.render()     
            
        # if max_episode_steps and elapsed_steps >= max_episode_steps:
        #     break
    agent.close()
    return episode_reward, elapsed_steps


if __name__ == '__main__':
    # 训练
    env = myenv.Myenv()
    print('load env sucess!!')
    agent = DQNAgent(env)
    episode_rewards = []
    for episode in range(1000):
        episode_reward, elapsed_steps = play_episode(env, agent, mode='train')
        episode_rewards.append(episode_reward)
        print('train episode %d: reward = %.2f, steps = %d' %
              (episode, episode_reward, elapsed_steps))

    plt.plot(episode_rewards)
    plt.show()
    plt.savefig('.//figure//iter.png')