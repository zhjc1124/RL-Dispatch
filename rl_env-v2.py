import torch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch.nn.functional as F

TIME_CONSTRAINS = np.load('./raw_data/dis_time.npy')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strftime(time):
    return time.strftime('%Y-%m-%d %H:%M:%S')

def station_onehot(station):
    return F.one_hot(torch.tensor(station).long(), 118)

class Dispatch:
    def __init__(self, data):
        self.sender_station = data['sender_station']
        self.receiver_station = data['receiver_station']
        self.send_step = data['send_step']
        # self.receive_step = data['receive_step']
        self.arrive_step = 0
        self.current_station = data['sender_station']
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
        self.states = torch.cat((
            station_onehot(self.current_station),
            torch.tensor([self.left_step*10])
            )).unsqueeze(0)
        self.subways = []
        self.actions = []
        self.action_probs = []
        self.action_steps = []
    

    def reward(self):
        reward = (len(self.hops) - 1) * self.hopcost
        if self.left_step > -1 and self.hops[-1] == self.receiver_station:
            reward += self.custpay
        return reward

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

    def record_states(self):
        state = torch.cat((                                     
            station_onehot(self.current_station),
            torch.tensor([self.left_step*10])
            )).unsqueeze(0)
        self.states = torch.cat((self.states, state))

    def output(self):
        if self.left_step < 0:
            lefttime = -1
        else:
            lefttime = self.left_step * 10

        return [self.hops[-1], self.receiver_station, lefttime]

class Myenv:
    def __init__(self):
        self.station_num = 118
        self.episode = timedelta(minutes=10)                       # one episode time 10 minutes
        self.time = None                                           # current time
        self.dispatchs = None                                      # total dispatchs
        self.subways = None                                        # total subways
        self.dispatchs_eval = None                                 # dispatchs per day
        self.subways_eval = None                                   # subways per day
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
        for i in range(len(actions)):
            self.dispatchs_waiting[i].step(actions[i], action_probs[i], self.step_nums)
            self.dispatchs_selected.append(self.dispatchs_waiting[i])
        self.dispatchs_waiting = []

        selected = []
        for dispatch in self.dispatchs_selected:
            avail_subway = self.subways_entering[self.subways_entering['swipe_in_station']==dispatch.hops[-2]]
            avail_subway = avail_subway[avail_subway['swipe_out_station']==dispatch.hops[-1]]
            if not avail_subway.empty:
                chosed_subway = avail_subway.sample().iloc[0]
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
        waiting = []
        for dispatch in self.dispatchs_delivering:
            dispatch.update()
            if dispatch.arrive_step+1 == self.step_nums:
                self.state['packages'][1][dispatch.hops[-2], dispatch.hops[-1]] -= 1
                if dispatch.hops[-1] == dispatch.receiver_station:
                    self.dispatchs_arrived.append(dispatch)
                else:
                    self.state['packages'][0][dispatch.hops[-1]] += 1
                    dispatch.record_states()
                    waiting.append(dispatch)
            else:
                delivering.append(dispatch)
        
        self.dispatchs_delivering = delivering

        arrived = self.subways_commuting[self.subways_commuting['swipe_out_step'] <= self.step_nums]
        self.subways_commuting = self.subways_commuting.drop(arrived.index)
        for index, row in arrived.iterrows():
            self.state['passengers'][1][row['swipe_in_station'], row['swipe_out_station']] -= 1

        dispatchs_handling = self.dispatchs_eval[self.dispatchs_eval['send_step']+1 <= self.step_nums]
        self.dispatchs_eval = self.dispatchs_eval.drop(dispatchs_handling.index)

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


        return self.dispatchs_waiting, self.step_nums == 144, self.time
    
    def reset(self, day=1):
        '''
        reset the environment.
        return states and time
        '''
        
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

        self.time = datetime(2021, 11, day, 0, 0, 0)

        self.dispatchs_eval = self.dispatchs.copy(deep=True)
        self.dispatchs_eval = self.dispatchs_eval[self.dispatchs_eval['send_datetime'].apply(lambda x: x.day) == day]
        self.dispatchs_eval['send_step'] = self.dispatchs_eval['send_datetime'].apply(self.time2step)

        # self.dispatchs_eval = self.dispatchs_eval.sample(n=10)
        # self.dispatchs_eval = self.dispatchs_eval.sort_values(by='send_datetime')
        # self.dispatchs_eval = self.dispatchs_eval.reset_index(drop=True)

        self.dispatchs_eval.to_csv('./dataset/dispatchs_eval.csv', sep=',')

        subways_eval = self.subways.copy(deep=True)
        subways_eval['swipe_in_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_in_time'])
        subways_eval['swipe_out_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_out_time'])
        subways_eval = subways_eval[subways_eval['swipe_in_time']<subways_eval['swipe_out_time']]
        subways_eval = subways_eval.sort_values(by='swipe_in_time')
        subways_eval = subways_eval.reset_index(drop=True)
        subways_eval['swipe_in_step'] = subways_eval['swipe_in_time'].apply(self.time2step)
        subways_eval['swipe_out_step'] = subways_eval['swipe_out_time'].apply(self.time2step)
        subways_eval.to_csv('./dataset/subways_eval.csv', sep=',')
        self.subways_eval = subways_eval

        self.step_nums = 0
        self.state['time'] = self.time

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

        self.dispatchs = dispatchs
        dispatchs.to_csv('./dataset/dispatchs_sorted.csv', sep=',')

        subways = pd.read_csv('./dataset/subways.csv', sep=',', index_col=0)
        subways = subways[subways['swipe_in_station']!=subways['swipe_out_station']]
        subways = subways.sort_values(by='swipe_in_time')
        subways = subways.reset_index(drop=True)        
        self.subways = subways

    def total_reward(self):
        total_reward = 0
        for dispatch in self.dispatchs_arrived:
            total_reward += dispatch.reward()
        for dispatch in self.dispatchs_selected:
            dispatch.hops.pop()
            total_reward += dispatch.reward()

if __name__ == '__main__':
    env = Myenv()
    dispatchs, done, _ = env.reset()
    while True:
        actions = []
        for dispatch in dispatchs:
            actions.append(dispatch .receiver_station)
        dispatchs, done, _ = env.step(actions)
        if done:

            print(env.total_reward())
            break