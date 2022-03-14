from glob import glob
import pickle
from collections import namedtuple
from itertools import count
from rl_env_DQN import Myenv, UPPERBOUND

import os, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import random

# Parameters
gamma = 0.9
render = False
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
log_interval = 10


GLOBAL_STATE_SIZE = 2 * (UPPERBOUND+1) * UPPERBOUND
COMMON_STATE_SIZE = UPPERBOUND * 2 + 3
INSTANT_STATE_SIZE = UPPERBOUND + 3

num_state = GLOBAL_STATE_SIZE + COMMON_STATE_SIZE + INSTANT_STATE_SIZE
num_action = UPPERBOUND


Transition = namedtuple('Transition', ['state', 'action',  'next_state', 'reward', 'dispatch', 'done'])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DELTA = torch.load('./dataset/delta.pth')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(2, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.conv3 = nn.Conv2d(10, 2, 5)

        self.conv1d1 = nn.Conv1d(1, 10, 3, 2)
        self.conv1d2 = nn.Conv1d(10, 20, 3, 2)

        self.fc1 = nn.Linear(740, 300)
        self.fc2 = nn.Linear(300, num_action)

    def forward(self, x):
        in_size = x.size(0)
        global_state = x[:, :GLOBAL_STATE_SIZE].view(in_size, 2, UPPERBOUND+1, UPPERBOUND)

        global_state = self.conv1(global_state)
        global_state = F.relu(global_state)
        global_state = F.max_pool2d(global_state, 2, 2)

        global_state = self.conv2(global_state)
        global_state = F.relu(global_state)
        global_state = F.max_pool2d(global_state, 2, 2)

        global_state = self.conv3(global_state)
        global_state = F.relu(global_state)
        global_state = F.max_pool2d(global_state, 2, 2)

        global_state = global_state.view(in_size, -1)

        private_state = x[:, GLOBAL_STATE_SIZE:]

        x = torch.cat((global_state, private_state), dim=1)

        x = x.view(in_size, 1, -1)
        x = self.conv1d1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2, 2)

        x = self.conv1d2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2, 2)

        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x)
        return action_value

class DQN():
    capacity = 20000
    learning_rate = 1e-2
    memory_count = 0
    batch_size = 32
    gamma = 0.9
    update_count = 0

    def __init__(self, mode='train'):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net().to(DEVICE), Net().to(DEVICE)
        self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')
        self.mode = mode

    def select_action(self, state):
        value = self.act_net(state)

        o = state[:, GLOBAL_STATE_SIZE:GLOBAL_STATE_SIZE+num_action].argmax(dim=1)
        d = state[:, GLOBAL_STATE_SIZE+num_action:GLOBAL_STATE_SIZE+2*num_action].argmax(dim=1)
        current_step = state[:, -1].long().detach().cpu()
        left_step = (state[:, -3]/10).long().detach().cpu()
        delta = DELTA[current_step, o, :UPPERBOUND]
        if (delta < left_step).any():
            mask = delta < left_step
        else:
            mask = delta < 10000

        if not mask.all():
            mask[:] = True
        mask = mask.to(DEVICE)
        masked_value = value * mask

        if self.mode == 'train':
            action_max_value, index = torch.max(masked_value, 1)
            action = index.item()
            if np.random.rand(1) >= 0.5: # epslion greedy
                action = np.random.choice(range(num_action), 1).item()
        else:
            if (delta > left_step).all():
                action = d
            else:
                action_max_value, index = torch.max(value, 1)
                action = index.item()
        return action

    def store_transition(self, info):
        state, action, next_state, reward, dispatch = info
        for i in range(len(state)):
            index = self.memory_count % self.capacity
            if next_state[i] is None:
                trans = Transition(state[i], action[i], state[i], reward[i], dispatch, True)
            else:
                trans = Transition(state[i], action[i], next_state[i], reward[i], dispatch, False)
            self.memory[index] = trans
            self.memory_count += 1

    def update(self):
        if self.memory_count >= self.capacity:
            state = torch.cat([t.state for t in self.memory]).float()
            action = torch.LongTensor([t.action for t in self.memory]).view(-1,1).long()
            reward = torch.tensor([t.reward for t in self.memory]).float()
            done = torch.BoolTensor([t.done for t in self.memory])
            next_state = torch.cat([t.next_state for t in self.memory]).float()
            reward = (reward - reward.mean()) / (reward.std() + 1e-7)


            #Update...
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                with torch.no_grad():
                    target_v = reward[index].to(DEVICE) + self.gamma * (self.target_net(next_state[index].to(DEVICE)).max(1)[0] * ~(done[index].to(DEVICE)))
                v = (self.act_net(state[index].to(DEVICE)).gather(1, action[index].to(DEVICE)))
                loss = self.loss_func(target_v.unsqueeze(1), v)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.update_count +=1
                if self.update_count % 3000 ==0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
        else:
            print("Memory Buff is too less")

    def save_param(self):
        net_file = './param/DQN/act_net' + str(time.time())[:10] + '.pkl'
        torch.save(self.act_net.state_dict(), net_file)
        return net_file

    def load_param(self, net_file):
        self.act_net.load_state_dict(torch.load(net_file))

def main():
    print('begin')
    torch.cuda.set_device(1)
    agent = DQN()
    for i_ep in range(1000):
        env = Myenv()
        dispatchs, done, _ = env.reset()
        for t in count(): 
            actions = []            
            for dispatch in dispatchs:
                state = env.get_state(dispatch).to(DEVICE)
                action= agent.select_action(state)
                actions.append(action)
                pass
            dispatchs, done, _ = env.step(actions)
            if done:
                dispatchs = env.dispatchs_arrived + env.dispatchs_selected + env.dispatchs_waiting
                for dispatch in dispatchs:
                    agent.store_transition(env.finish(dispatch))
                    pass
                agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
                # print(agent.memory_count)
                if agent.memory_count >= agent.capacity:
                    agent.update()
                    if i_ep % 10 == 0:
                        print("episodes {}, step is {} ".format(i_ep, t))
                        net_path = agent.save_param()
                        test(net_path, i_ep)
                break

def test(net_path, i_ep=0):
    torch.cuda.set_device(1)
    agent = DQN(mode='test')
    agent.load_param(net_path)
    env = Myenv()
    dispatchs, done, _ = env.reset()
    for t in count(): 
        actions = []            
        for dispatch in dispatchs:
            state = env.get_state(dispatch).to(DEVICE)
            action= agent.select_action(state)
            actions.append(action)
            pass
        dispatchs, done, _ = env.step(actions)
        if done:
            profit = env.total_profit()
            agent.writer.add_scalar('loss/profit', profit, global_step=i_ep)
            print(profit)
            print(len(env.dispatchs_arrived))
            print(env.evaluate())
            break
if __name__ == '__main__':
    main()
    # test('param/DQN/act_net1647418732.pkl')