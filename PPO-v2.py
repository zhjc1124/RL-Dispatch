import pickle
from collections import namedtuple
from itertools import count
from rl_env_v2 import Myenv

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
gamma = 0.99
render = False
seed = 1
log_interval = 10

env = Myenv()
num_state = 28444
num_action = 118
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'dispatch'])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DELTA = torch.load('./dataset/delta.pth')


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 5000)
        self.action_head = nn.Linear(5000, num_action)

    def forward(self, x):
        state = x

        x = F.relu(self.fc1(x))

        # if self.mode == 'train':  
        o = state[0, 28084:28084+118].argmax()  
        # d = state[0, 28084+118:28084+2*118].argmax()
        current_step = int(state[0, -1])
        left_step = int(state[0, -3]/10)
        delta = DELTA[current_step, o]
        if (delta < left_step).all():
            mask = delta < left_step
        else:
            mask = delta < 10000
        if not mask.all():
            mask[:] = True
        x = torch.exp(self.action_head(x)) * mask
        action_prob = x/x.sum()
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 5000)
        self.state_value = nn.Linear(5000, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self, mode='train'):
        super(PPO, self).__init__()
        self.actor_net = Actor().to(DEVICE)
        self.critic_net = Critic().to(DEVICE)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.mode = mode
        self.writer = SummaryWriter('./exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')

    def select_action(self, state):
        with torch.no_grad():
            action_prob = self.actor_net(state)

        if self.mode == 'train':
            c = Categorical(action_prob)
            action = c.sample()
        else:
            action = action_prob.argmax()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), './param/net_param/actor_net' + str(time.time())[:10] + '.pkl')
        torch.save(self.critic_net.state_dict(), './param/net_param/critic_net' + str(time.time())[:10] + '.pkl')

    def store_transition(self, info):
        state, action, action_prob, reward, dispatch = info
        for i in range(len(state)):
            trans = Transition(state[i], action[i], action_prob[i], reward[i], dispatch)
            self.buffer.append(trans)
            self.counter += 1


    def update(self, i_ep):
        state = torch.cat([t.state for t in self.buffer])
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

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
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
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
        dispatchs, done, _ = env.reset()
        for t in count():  
            actions = []
            action_probs = []
            for dispatch in dispatchs:
                state = env.get_state(dispatch)
                action, action_prob = agent.select_action(state)
                actions.append(action)
                action_probs.append(action_prob)
                pass
            dispatchs, done, _ = env.step(actions, action_probs)

            # if env.dispatchs_arrived:
            if done:
                print(env.total_reward())
                dispatchs = env.dispatchs_arrived + env.dispatchs_selected
                random.shuffle(dispatchs)
                for dispatch in dispatchs:
                    agent.store_transition(env.finish(dispatch))
                    pass
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                    agent.save_param()
                    agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
                

if __name__ == '__main__':
    main()
    print("end")
