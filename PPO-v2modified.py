import pickle
from collections import namedtuple
from itertools import count
from rl_env_v2modified import Myenv, UPPERBOUND

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


Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'dispatch'])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DELTA = torch.load('./dataset/delta.pth')


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

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
        action_prob = F.softmax(self.fc2(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(2, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.conv3 = nn.Conv2d(10, 2, 5)

        self.conv1d1 = nn.Conv1d(1, 10, 3, 2)
        self.conv1d2 = nn.Conv1d(10, 20, 3, 2)

        self.fc1 = nn.Linear(740, 300)
        self.fc2 = nn.Linear(300, 1)

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
        value = self.fc2(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 5
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
        self.writer = SummaryWriter('./expmodified')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 1e-3)
        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')

    # def select_action(self, state):
    #     with torch.no_grad():
    #         action_prob = self.actor_net(state)
    #
    #     if torch.isnan(action_prob).any():
    #         pass
    #     # d = state[:, 28084 + 118:28084 + 2 * 118].argmax(dim=1)
    #     if self.mode == 'train':
    #         if torch.isnan(action_prob).any():
    #             pass
    #         c = Categorical(action_prob)
    #         action = c.sample()
    #     else:
    #         action = action_prob.argmax()
    #     return action.item(), action_prob[:, action.item()].item()

    def select_action(self, state):
        with torch.no_grad():
            action_prob = self.actor_net(state)
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
        masked_prob = action_prob * mask
        masked_prob = masked_prob/masked_prob.sum(dim=1, keepdim=True)

        if self.mode == 'train':
            c = Categorical(masked_prob)
            action = c.sample()
        else:
            if (delta > left_step).all():
                action = d
            else:
                action = action_prob.argmax()
        return action.item(), action_prob[:, action.item()].item()
        # if random.randint(0, 5) == 0:
        #     return action.item(), action_prob[:, action.item()].item()
        # else:
        #     return int(d.item()), action_prob[:, int(d.item())].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        actor_file = './param/net_parammodified/actor_net' + str(time.time())[:10] + '.pkl'
        critic_file = './param/net_parammodified/critic_net' + str(time.time())[:10] + '.pkl'
        torch.save(self.actor_net.state_dict(), actor_file)
        torch.save(self.critic_net.state_dict(), critic_file)
        return actor_file, critic_file

    def load_param(self, params_file):
        actor_file,critic_file = params_file
        self.actor_net.load_state_dict(torch.load(actor_file))
        self.critic_net.load_state_dict(torch.load(critic_file))

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
                Gt_index = Gt[index].view(-1, 1).to(DEVICE)
                V = self.critic_net(state[index].to(DEVICE))
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_probs = self.actor_net(state[index].to(DEVICE))

                action_prob = action_probs.gather(1, action[index].to(DEVICE)) # new policy
                entropy = Categorical(action_probs).entropy()

                ratio = (action_prob/old_action_log_prob[index].to(DEVICE))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2) - entropy * 0.1  # MAX->MIN desent
                action_loss = action_loss.mean()
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
    torch.cuda.set_device(1)
    print('begin train')
    agent = PPO(mode='train')
    agent.load_param(('param/net_param/actor_net1647544284.pkl', 'param/net_param/critic_net1647544284.pkl'))
    for i_epoch in range(1000):
        env = Myenv()
        dispatchs, done, _ = env.reset()
        for t in count():  
            actions = []
            action_probs = []
            for dispatch in dispatchs:
                state = env.get_state(dispatch).to(DEVICE)
                action, action_prob = agent.select_action(state)
                actions.append(action)
                action_probs.append(action_prob)
                pass
            dispatchs, done, _ = env.step(actions, action_probs)

            # if env.dispatchs_arrived:
            if done:
                dispatchs = env.dispatchs_arrived + env.dispatchs_selected + env.dispatchs_waiting
                # random.shuffle(dispatchs)
                for dispatch in dispatchs:
                    agent.store_transition(env.finish(dispatch))
                    pass
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                    if (i_epoch+1) % 20 == 0:
                        params_file = agent.save_param()
                        test(params_file, i_epoch)
                    agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
                
def test(params_file, i_epoch):
    torch.cuda.set_device(1)
    print('begin test')
    agent = PPO(mode='test')
    agent.load_param(params_file)
    env = Myenv()
    dispatchs, done, _ = env.reset()
    for t in count():  
        actions = []
        action_probs = []
        for dispatch in dispatchs:
            state = env.get_state(dispatch).to(DEVICE)
            action, action_prob = agent.select_action(state)
            actions.append(action)
            action_probs.append(action_prob)
            pass
        dispatchs, done, _ = env.step(actions, action_probs)
        if done:
            profit = env.total_profit()
            agent.writer.add_scalar('loss/profit', profit, global_step=i_epoch)
            print(profit)
            print(len(env.dispatchs_arrived))
            print(env.evaluate())
            break

if __name__ == '__main__':
    main()
    print("end")
    # test(('param/net_param/actor_net1647524269.pkl', 'param/net_param/critic_net1647524269.pkl'), 0)
