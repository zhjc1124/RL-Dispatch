import copy
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import myenv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class Net(nn.Module):
#     def __init__(self):
#         nn.Module.__init__(self)
#         self.input_layer = nn.Sequential(nn.Linear(3 + 118*119*2, 2048),
#                                 nn.Dropout(0.5),
#                                 nn.ReLU())
#         self.hidden_layer1 = nn.Sequential(nn.Linear(2048, 1024),
#                                 nn.Dropout(0.3),
#                                 nn.ReLU())
#         self.hidden_layer2 = nn.Sequential(nn.Linear(1024, 1024),
#                                 # nn.Dropout(0.3),
#                                 nn.ReLU())
#         self.hidden_layer3 = nn.Sequential(nn.Linear(1024, 2048),
#                                 # nn.Dropout(0.3),
#                                 nn.ReLU())
#         self.output_layer = nn.Linear(2048, 118)
        
#     def forward(self, x):
#         x = self.input_layer(x)
#         x = self.hidden_layer1(x)
#         x = self.hidden_layer2(x)
#         x = self.hidden_layer3(x)
#         x = self.output_layer(x)
        
#         return x
# net = Net()

def observationEncode(env_state):
    ''' 状态编码
        size(len(dispatchs), 3+118*119*2), 
          packages: [size(1, 118), size(118, 118)], 
          passengers: [size(1, 118), size(118, 118)], 
        全部展平后编码
    '''
    packages_tensor_0 = torch.as_tensor(env_state['packages'][0])
    packages_tensor_1 = env_state['packages'][1].view(-1).clone().detach()
    packages_tensor = torch.cat((packages_tensor_0.unsqueeze(0), packages_tensor_1.unsqueeze(0)), 1)

    passengers_tensor_0 = torch.as_tensor(env_state['passengers'][0])
    passengers_tensor_1 = env_state['passengers'][1].view(-1).clone().detach()
    passengers_tensor = torch.cat((passengers_tensor_0.unsqueeze(0), passengers_tensor_1.unsqueeze(0)), 1)
    p_tensor = torch.cat((packages_tensor, passengers_tensor), 1)
    
    dispatchs_tensor = torch.as_tensor(env_state['dispatchs'])
    if dispatchs_tensor.numel():
        row_num = dispatchs_tensor.size()[0]
        p_m_tensor = p_tensor.repeat([row_num, 1])
        return torch.cat((dispatchs_tensor, p_m_tensor), 1)
    else: 
        return torch.cat((torch.ones(1,3) * -1.0, p_tensor),1)


env = myenv.Myenv()
state = env.reset()

class DQNReplayer:
    def __init__(self, capacity):
        # 记忆池用DataFrame格式储存
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['state', 'action', 'reward', 'next_state', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity                # 记忆池的容量

    # 储存经验，超过容量重新填充
    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity                       
        self.count = min(self.count + 1, self.capacity)

    # 经验抽取
    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

class DQNAgent:
    def __init__(self, env):
        self.action_n = 118                                                                           # 来自环境的动作数
        self.gamma = 0.99

        # 容量capacity定义10000
        self.replayer = DQNReplayer(10000)

        ### 输入状态得到输出动作------评价网络
        self.evaluate_net = self.build_net(
                input_size = 3 + env.station_num * (env.station_num + 1) *2,                                                       # dispatch, packages, passengers
                hidden_sizes=[2048, 1024, 1024], output_size = self.action_n)                       # 来自环境的状态数
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

        ### 输入状态得到输出动作
    def build_net(self, input_size, hidden_sizes, output_size):
        layers = []
        for input_size, output_size in zip(
                [input_size,] + hidden_sizes, hidden_sizes + [output_size,]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.Dropout(0.3))
            layers.append(nn.ReLU())
        layers = layers[:-2]
        model = nn.Sequential(*layers)
        return model

    
    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []
            self.target_net = copy.deepcopy(self.evaluate_net)

    # agent决策
    def step(self, observation, reward, done):
        if self.mode == 'train' and np.random.rand() < 0.01:
            # epsilon-greedy policy in train mode
            action = np.random.randint(self.action_n)
        else:
            state_tensor = observation                                  # get state
            state_tensor_device = state_tensor.to(device)
            q_tensor = self.evaluate_net(state_tensor_device)
            action_tensor = torch.argmax(q_tensor)                      # 选择最大的Q值对应的动作----------此动作是针对dispatchs
            action = action_tensor.item()                               # 选择最大的Q值----张量的元素值
            '''
                应该有个动作过滤器(ETA过滤)，消除无意义的动作，加快收敛
            '''
            if observation[2] < 0:
                action = observation[1].item()                          # 如果超时选择直达
        if self.mode == 'train':
            self.trajectory += [observation, reward, done, action]      # 记录当前的state、reward、done、action
            if len(self.trajectory) >= 8:
                state, _, _, act, next_state, reward, done, _ = \
                        self.trajectory[-8:]
                self.replayer.store(state, act, reward, next_state, done)
            if self.replayer.count >= self.replayer.capacity * 0.95:
                    # skip first few episodes for speed
                self.learn()                                            # 开始学习
        return action

    def close(self):
        pass

    def learn(self):
        # replay
        states, actions, rewards, next_states, dones = \
                self.replayer.sample(1024) # replay transitions                             # 其中states 的长度是动态的
        state_tensor = torch.as_tensor(states, dtype=torch.float)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float)
        done_tensor = torch.as_tensor(dones, dtype=torch.float)

        # train
        next_state_tensor_device = next_state_tensor.to(device)
        next_q_tensor = self.target_net(next_state_tensor_device)
        next_max_q_tensor, _ = next_q_tensor.max(axis=-1)
        target_tensor = reward_tensor.to(device) + self.gamma * (1. - done_tensor.to(device)) * next_max_q_tensor.to(device)
        state_tensor_device = state_tensor.to(device)
        pred_tensor = self.evaluate_net(state_tensor_device)
        action_tensor = action_tensor.to(device)
        q_tensor = pred_tensor.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        loss_tensor = self.loss(target_tensor, q_tensor)
        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()


agent = DQNAgent(env)
agent.evaluate_net.to(device)


def play_episode(env, agent, max_episode_steps=None, mode=None, render=False):
    observation, reward, done = env.reset(), 0., False                                  ## 来自环境的reset函数
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        if done:
            break
        state_tensor = observationEncode(observation)
        actions = []
        for i in range(state_tensor.size(0)):
            if state_tensor[i][0] >=0:
                actions.append(agent.step(state_tensor[i], reward, done))
        if render:
            env.render()
        print(actions)                                                              
        observation = env.step(actions)
        reward = observation['reward']
        done = observation['done']
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps >= max_episode_steps:
            break
    agent.close()
    return episode_reward, elapsed_steps

'''绘制网络结构图'''

# from torchviz import make_dot
# model=agent.evaluate_net
# y=model(torch.rand(3+118*119*2).to(device))
# g = make_dot(y)
# g.render('espnet_model', view=True)




print('============ train ============')
env = myenv.Myenv()

episode_rewards = []
for episode in range(10000):
    episode_reward, elapsed_steps = play_episode(env, agent, mode='train')                  ## 来自环境的_max_episode_steps参数
    episode_rewards.append(episode_reward)
    print('train episode %d: reward = %.2f, steps = %d'%
            (episode, episode_reward, elapsed_steps))
plt.plot(episode_rewards)


