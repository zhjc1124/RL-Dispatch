import torch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch.nn.functional as F
import random

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

station_distances = np.load('./raw_data/station_distances.npy')
TIME_CONSTRAINS = np.load('./raw_data/dis_time.npy')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPPERBOUND = 118
DISPATCH_NUMS = 3000

def strftime(time):
    return time.strftime('%Y-%m-%d %H:%M:%S')

def station_onehot(station):
    return F.one_hot(torch.tensor(station).long(), UPPERBOUND)

class Dispatch:
    def __init__(self, data):
        self.status = 'waiting'
        self.sender_station = data['sender_station']
        self.receiver_station = data['receiver_station']
        self.send_step = data['send_step']
        # self.receive_step = data['receive_step']
        self.arrive_step = 0
        # self.current_station = data['sender_station']
        assert(self.sender_station!=self.receiver_station)
        self.time_constrain = TIME_CONSTRAINS[self.sender_station, self.receiver_station]
        self.left_step = self.time_constrain * 6     # 1 step = 10 min
        self.custpay = 5
        self.hopcost = -1
        self.hops = [self.sender_station]
        self.commons = torch.cat((
            station_onehot(self.sender_station), 
            station_onehot(self.receiver_station), 
            torch.tensor([self.time_constrain*60, self.custpay, self.hopcost])
            ))
        self.states = None
        self.subways = []
        self.actions = []
        self.action_probs = []
        self.action_steps = []
        self.rewards = []
        self.arrive_step = -1

    def profit(self):
        profit = (len(self.hops) - 1) * self.hopcost
        if self.left_step > -1 and self.status == 'arrived':
            profit += self.custpay        
        return profit

    def deliver(self, chosed_subway):
        self.subways.append(chosed_subway)
        self.arrive_step = chosed_subway['swipe_out_step']

    def step(self, action, action_prob, action_step):
        self.hops.append(action)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.action_steps.append(action_step)

    def update(self):
        self.left_step -= 1

    def record_states(self, current_step):
        state = torch.cat((                                     
            station_onehot(self.hops[-1]),
            torch.tensor([self.left_step*10, len(self.hops) - 1, current_step])
            )).unsqueeze(0)
        if self.states is None:
            self.states = state
        else:
            self.states = torch.cat((self.states, state))

    # def record_rewards(self):
    #     reward = 0
    #     if self.left_step > -1 and self.hops[-1] == self.receiver_station:
    #         reward += 5                                  # ????????????
    #     return reward 
    
    def record_rewards(self):
        reward = -(len(self.hops) - 1) * 10                  # ????????????
        if self.left_step < 0:
            reward += self.left_step * 10                       # ??????????????????

        if self.hops[-1] == self.receiver_station:
            reward += 100                                    # ????????????

        last_station = None
        if self.status == 'selected':
            current_station = self.hops[-2]
            if len(self.hops) >= 3:
                last_station = self.hops[-3]
        else:
            current_station = self.hops[-1]
            last_station = self.hops[-1]
        receiver_station = self.receiver_station
<<<<<<< HEAD
        if last_station and current_station != receiver_station:
=======
        if current_station != receiver_station:
>>>>>>> 1cfb87d8fab459b13c0b834007638151045a3a33
            current_distance = station_distances[current_station, receiver_station]
            last_distance = station_distances[last_station, receiver_station]
            if current_distance > last_distance:
                reward += (last_distance - current_distance) * 10  # ????????????
        self.rewards.append(reward)
        return reward 

    def get_state(self):
        return torch.cat((self.commons, self.states[-1]))
    

class Myenv:
    def __init__(self, day=1):
        self.station_num = UPPERBOUND
        self.day = day
        self.episode = timedelta(minutes=10)                       # one episode time 10 minutes
        self.time = datetime(2021, 11, self.day, 0, 0, 0)          # current time
        self.dispatchs = None                                      # total dispatchs
        self.subways = None                                        # total subways

        self.dispatchs_eval_bak = None
        self.subways_eval_bak = None

        self.load_dataset()

        self.dispatchs_waiting = []
        self.dispatchs_selected = []
        self.dispatchs_delivering = []
        self.dispatchs_arrived = []
        self.infos = []

    def time2step(self, x):
        return (x-self.time).seconds // 600

    def step(self, actions, action_probs=None):
        # chose action for packages
        assert(len(actions) == len(self.dispatchs_waiting))
        if action_probs is None:
            action_probs = [1] * len(actions)

        waiting = []
        for i in range(len(actions)):
            if actions[i] == self.dispatchs_waiting[i].hops[-1]:
                waiting.append(self.dispatchs_waiting[i])
            else:
                self.dispatchs_waiting[i].step(actions[i], action_probs[i], self.step_nums)
                self.dispatchs_waiting[i].status = 'selected'
                self.dispatchs_selected.append(self.dispatchs_waiting[i])
        self.dispatchs_waiting = waiting

        selected = []
        for dispatch in self.dispatchs_selected:
            avail_subway = self.subways_entering[self.subways_entering['swipe_in_station']==dispatch.hops[-2]]
            avail_subway = avail_subway[avail_subway['swipe_out_station']==dispatch.hops[-1]]
            if not avail_subway.empty:
                chosed_subway = avail_subway.sample().iloc[0]
                dispatch.status = 'delivering'
                dispatch.deliver(chosed_subway)
                self.dispatchs_delivering.append(dispatch)
                self.state['packages'][0][dispatch.hops[-2]] -= 1
                self.state['packages'][1][dispatch.hops[-2], dispatch.hops[-1]] += 1
            else:
                selected.append(dispatch)
        self.dispatchs_selected = selected

        # set the commuting passengers
        for index, row in self.subways_entering.iterrows():
            self.state['passengers'][0][row['swipe_in_station']] -= 1
            self.state['passengers'][1][row['swipe_in_station'], row['swipe_out_station']] += 1

        # self.subways_commuting = self.subways_commuting.append(self.subways_entering)
        self.subways_commuting = pd.concat([self.subways_commuting, self.subways_entering])
        # next episode
        self.time += self.episode
        self.step_nums += 1

        for dispatch in self.dispatchs_selected:
            dispatch.update()

        delivering = []
        waiting = self.dispatchs_waiting
        for dispatch in self.dispatchs_delivering:
            dispatch.update()
            if dispatch.arrive_step+1 == self.step_nums:
                self.state['packages'][1][dispatch.hops[-2], dispatch.hops[-1]] -= 1
                dispatch.record_rewards()
                if dispatch.hops[-1] == dispatch.receiver_station:
                    dispatch.status = 'arrived'
                    self.arrive_step = self.step_nums
                    self.dispatchs_arrived.append(dispatch)
                else:
                    self.state['packages'][0][dispatch.hops[-1]] += 1
                    dispatch.record_states(self.step_nums)
                    dispatch.status = 'waiting'
                    waiting.append(dispatch)
            else:
                delivering.append(dispatch)
        
        self.dispatchs_delivering = delivering
        self.dispatchs_waiting = waiting

        arrived = self.subways_commuting[self.subways_commuting['swipe_out_step'] <= self.step_nums]
        self.subways_commuting = self.subways_commuting.drop(arrived.index)
        for index, row in arrived.iterrows():
            self.state['passengers'][1][row['swipe_in_station'], row['swipe_out_station']] -= 1

        if self.step_nums == 143:
            return self.dispatchs_waiting, self.step_nums == 143, self.time

        dispatchs_handling = self.dispatchs_eval[self.dispatchs_eval['send_step']+1 <= self.step_nums]
        # print('in:', len(dispatchs_handling))
        self.dispatchs_eval = self.dispatchs_eval.drop(dispatchs_handling.index)

        waiting = self.dispatchs_waiting
        for index, row in dispatchs_handling.iterrows():
            dispatch = Dispatch(row)
            dispatch.record_states(self.step_nums)
            waiting.append(dispatch)
            self.state['packages'][0][dispatch.sender_station] += 1
        self.dispatchs_waiting = waiting

        self.subways_entering = self.subways_eval[self.subways_eval['swipe_in_step'] <= self.step_nums]
        self.subways_eval = self.subways_eval.drop(self.subways_entering.index)
        for index, row in self.subways_entering.iterrows():
            self.state['passengers'][0][row['swipe_in_station']] += 1
        info = torch.zeros(2, self.station_num+1, self.station_num)
        info[0, 0] = self.state['packages'][0]
        info[0, 1:] = self.state['packages'][1]
        info[1, 0] = self.state['passengers'][0]
        info[1, 1:] = self.state['passengers'][1]
        self.infos.append(info)
        # print('all:', len(self.dispatchs_waiting+self.dispatchs_delivering+self.dispatchs_selected+self.dispatchs_arrived))
        return self.dispatchs_waiting, self.step_nums == 143, self.time
    
    def reset(self):
        '''
        reset the environment.
        return states and time
        '''
        seed = 10
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)

        self.subways_eval = self.subways_eval_bak.copy(deep=True)
        self.dispatchs_eval = self.dispatchs_eval_bak.copy(deep=True)

        self.dispatchs_waiting = []
        self.dispatchs_selected = []
        self.dispatchs_delivering = []
        self.dispatchs_arrived = []
        self.infos = []

        self.time = datetime(2021, 11, self.day, 0, 0, 0)

        self.subways_entering = None
        self.subways_commuting = pd.DataFrame(None,columns=['card_id','swipe_in_time','swipe_in_station','swipe_out_time','swipe_out_station'])
        self.step_nums = 0
        self.state = {
            'dispatchs': [],
            'packages': [
                torch.zeros(self.station_num),                     # waiting packages
                torch.zeros(self.station_num, self.station_num)    # delivering packages
            ],
            'passengers': [
                torch.zeros(self.station_num),                     # entering passengers
                torch.zeros(self.station_num, self.station_num)    # commuting passengers
            ],
            'done': False,
            'time': self.time,
            'reward': 0
        }

        dispatchs_handling = self.dispatchs_eval[self.dispatchs_eval['send_step']+1 <= self.step_nums]
        self.dispatchs_eval = self.dispatchs_eval.drop(dispatchs_handling.index)
        waiting = []
        for index, row in dispatchs_handling.iterrows():
            dispatch = Dispatch(row)
            waiting.append(dispatch)
            self.state['packages'][0][dispatch.sender_station] += 1
        self.dispatchs_waiting = waiting

        self.subways_entering = self.subways_eval[self.subways_eval['swipe_in_step'] <= self.step_nums]
        self.subways_eval = self.subways_eval.drop(self.subways_entering.index)
        for index, row in self.subways_entering.iterrows():
            self.state['passengers'][0][row['swipe_in_station']] += 1
        info = torch.zeros(2, self.station_num+1, self.station_num)
        info[0, 0] = self.state['packages'][0]
        info[0, 1:] = self.state['packages'][1]
        info[1, 0] = self.state['passengers'][0]
        info[1, 1:] = self.state['passengers'][1]
        self.infos.append(info)
        return self.dispatchs_waiting, False, self.time

    def load_dataset(self):

        dispatchs = pd.read_csv('./dataset/dispatchs.csv', sep=',', index_col=0)
        dispatchs = dispatchs[dispatchs['sender_station'] != dispatchs['receiver_station']]
        dispatchs['send_datetime'] = pd.to_datetime(dispatchs['send_datetime'])
        dispatchs['receive_datetime'] = pd.to_datetime(dispatchs['receive_datetime'])
        dispatchs = dispatchs[dispatchs['send_datetime'] < dispatchs['receive_datetime']]
        dispatchs = dispatchs[dispatchs['send_datetime'].apply(lambda x: x.day) == dispatchs['receive_datetime'].apply(lambda x: x.day)]
        dispatchs = dispatchs.sort_values(by='send_datetime')
        dispatchs = dispatchs.reset_index(drop=True)

        # dispatchs.to_csv('./dataset/dispatchs_sorted.csv', sep=',')

        subways = pd.read_csv('./dataset/subways.csv', sep=',', index_col=0)
        subways = subways[subways['swipe_in_station']!=subways['swipe_out_station']]
        subways = subways.sort_values(by='swipe_in_time')
        subways = subways.reset_index(drop=True)

        dispatchs_eval = dispatchs.copy(deep=True)
        dispatchs_eval = dispatchs_eval[dispatchs_eval['send_datetime'].apply(lambda x: x.day) == self.day]
        dispatchs_eval['send_step'] = dispatchs_eval['send_datetime'].apply(self.time2step)
        dispatchs_eval['receive_step'] = dispatchs_eval['receive_datetime'].apply(self.time2step)

        dispatchs_eval = dispatchs_eval[dispatchs_eval['sender_station'] < UPPERBOUND]
        dispatchs_eval = dispatchs_eval[dispatchs_eval['receiver_station'] < UPPERBOUND]

        dispatchs_eval = dispatchs_eval.sample(n=DISPATCH_NUMS)     # 2000, 7438
        # dispatchs_eval = dispatchs_eval.sample(frac=0.8) # 31312(0.8), 395(0.01, 274)
        dispatchs_eval = dispatchs_eval.sort_values(by='send_datetime')
        dispatchs_eval = dispatchs_eval.reset_index(drop=True)

        dispatchs_eval.to_csv('./dataset/dispatchs_eval.csv', sep=',')
        self.dispatchs_eval_bak = dispatchs_eval

        subways_eval = subways.copy(deep=True)
        subways_eval['swipe_in_time'] = pd.to_datetime(f'2021-11-{self.day:02d} ' + subways_eval['swipe_in_time'])
        subways_eval['swipe_out_time'] = pd.to_datetime(f'2021-11-{self.day:02d} ' + subways_eval['swipe_out_time'])
        subways_eval = subways_eval[subways_eval['swipe_in_time']<subways_eval['swipe_out_time']]
        subways_eval = subways_eval.sort_values(by='swipe_in_time')
        subways_eval = subways_eval.reset_index(drop=True)
        subways_eval['swipe_in_step'] = subways_eval['swipe_in_time'].apply(self.time2step)
        subways_eval['swipe_out_step'] = subways_eval['swipe_out_time'].apply(self.time2step)

        subways_eval = subways_eval[subways_eval['swipe_in_station'] < UPPERBOUND]
        subways_eval = subways_eval[subways_eval['swipe_out_station'] < UPPERBOUND]

        subways_eval.to_csv('./dataset/subways_eval.csv', sep=',')
        self.subways_eval_bak = subways_eval

    def total_profit(self):
        total_profit = 0
        for dispatch in self.dispatchs_arrived + self.dispatchs_selected + self.dispatchs_waiting:
            total_profit += dispatch.profit()
        return total_profit
    
    def get_state(self, dispatch):
        global_state = self.infos[-1]
        private_state = dispatch.get_state()
        state = torch.cat((global_state.flatten(), private_state))
        state = state.float().unsqueeze(0)
        return state

    def finish(self, dispatch):
        n = len(dispatch.action_steps)
        states = []
        actions = dispatch.actions
        for i in range(n):
            global_state = self.infos[dispatch.action_steps[i]]
            private_state = torch.cat((dispatch.commons, dispatch.states[i]))
            state = torch.cat((global_state.flatten(), private_state))
            state = state.float().unsqueeze(0)
            states.append(state)

        # rewards = [0] * n
        # rewards[-1] = dispatch.reward()
        if dispatch.status == 'selected':
            dispatch.record_rewards()
        rewards = dispatch.rewards
        old_action_log_probs = dispatch.action_probs
        return states, actions, old_action_log_probs, rewards, dispatch

    def evaluate(self):
        arrived = 0
        hop_num = 0
        for d in self.dispatchs_arrived:
            if d.status == 'arrived' and d.left_step >= 0:
                arrived += 1
            hop_num += len(d.hops) - 1
        profit_rate = self.total_profit()/DISPATCH_NUMS
        deliver_rate = arrived/DISPATCH_NUMS
        average_hop = hop_num/len(self.dispatchs_arrived)
        return profit_rate, deliver_rate, average_hop

        

if __name__ == '__main__':
    env = Myenv()
    dispatchs, done, _ = env.reset()
    while True:
        actions = []
        for dispatch in dispatchs:
            actions.append(dispatch.receiver_station)
        dispatchs, done, _ = env.step(actions)
        if done:
            print(env.total_profit())
            print(len(env.dispatchs_arrived))
            print(env.evaluate())
            # import pickle
            # with open('./analyze/direct.pkl', 'wb') as f:
            #     pickle.dump(env, f)
