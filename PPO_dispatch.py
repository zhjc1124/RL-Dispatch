import pickle
from collections import namedtuple
from itertools import count
from rl_env import Myenv

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

# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10

env = Myenv()
num_state = 10000
num_action = 118
torch.manual_seed(seed)
Transition = namedtuple('Transition', ['state', 'actions',  'a_log_probs', 'reward', 'next_state'])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(2, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 10, 3)
        self.fc1 = nn.Linear(10*26*26, 800)
        self.fc2 = nn.Linear(800+1, 300)
        self.action_head = nn.Linear(300, 118)

    def forward(self, x0, lefttime):
        x0 = x0.unsqueeze(0)
        x0 = self.conv1(x0)
        x0 = F.relu(x0)
        x0 = F.max_pool2d(x0, 2, 2)
        x0 = self.conv2(x0)
        x0 = F.relu(x0)
        x0 = F.max_pool2d(x0, 2, 2)
        x0 = self.conv3(x0) 
        x0 = F.relu(x0)
        x0 = x0.view(1, -1)
        x0 = self.fc1(x0)
        x0 = F.relu(x0)
        x = torch.cat((x0, lefttime.view(1,-1)), 1)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob



class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(2, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 10, 3)
        self.fc1 = nn.Linear(10*26*26, 800)
        self.fc2 = nn.Linear(800+1, 300)
        self.value_head = nn.Linear(300, 1)

    def forward(self, x0, lefttime):
        x0 = x0.unsqueeze(0)
        x0 = self.conv1(x0)
        x0 = F.relu(x0)
        x0 = F.max_pool2d(x0, 2, 2)
        x0 = self.conv2(x0)
        x0 = F.relu(x0)
        x0 = F.max_pool2d(x0, 2, 2)
        x0 = self.conv3(x0) 
        x0 = F.relu(x0)
        x0 = x0.view(1, -1)
        x0 = self.fc1(x0)
        x0 = F.relu(x0)
        x = torch.cat((x0, lefttime.view(1,-1)), 1)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.value_head(x)
        return value

class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().to(DEVICE)
        self.critic_net = Critic().to(DEVICE)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('../exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        dispatchs = state[0]
        info = state[1]
        actions = []
        action_probs = []
        with torch.no_grad():
            for d in dispatchs:
                source = F.one_hot(d[0].long(), 118)
                target = F.one_hot(d[1].long(), 118)
                lefttime = d[2]
                station_info = torch.cat((source.unsqueeze(0), target.unsqueeze(0)), 0)
                x0 = torch.cat((station_info.unsqueeze(1), info), 1)
                action_prob = self.actor_net(x0, lefttime)
                c = Categorical(action_prob)
                action = c.sample()
                actions.append(action.item())
                action_probs.append(action_prob[:,action.item()].item())
        return actions, action_probs

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1


    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        actions = torch.tensor([t.actions for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_probs = torch.tensor([t.a_log_probs for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = 0
                state_index = state[index]

                dispatchs = state[0]
                info = state[1]
                for d in dispatchs:
                    source = F.one_hot(d[0].long(), 118)
                    target = F.one_hot(d[1].long(), 118)
                    lefttime = d[2]
                    station_info = torch.cat((source.unsqueeze(0), target.unsqueeze(0)), 0)
                    x0 = torch.cat((station_info.unsqueeze(1), info), 1)
                    V = self.critic_net(x0, lefttime)

                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!

                i=0
                action_prob = []
                for d in dispatchs:
                    source = F.one_hot(d[0].long(), 118)
                    target = F.one_hot(d[1].long(), 118)
                    lefttime = d[2]
                    station_info = torch.cat((source.unsqueeze(0), target.unsqueeze(0)), 0)
                    x0 = torch.cat((station_info.unsqueeze(1), info), 1)
                    action_prob.append(self.actor_net(x0, lefttime).gather(1, actions[index][i]))
                    i += 1 

                ratio = torch.mean(action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience

    
def main():
    agent = PPO()
    for i_epoch in range(1000):
        state = env.reset()

        for t in count():
            if len(state[0]) == 0:
                state, reward, done, _ = env.step([])
                continue
            actions, action_probs = agent.select_action(state)
            next_state, reward, done, _ = env.step(actions)
            trans = Transition(state, actions, action_probs, reward, next_state)
            agent.store_transition(trans)
            state = next_state

            if done:
                if len(agent.buffer) >= agent.batch_size:agent.update(i_epoch)
                agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break

if __name__ == '__main__':
    main()
    print("end")