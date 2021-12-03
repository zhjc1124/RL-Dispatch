import torch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

TIME_CONSTRAINS = np.load('./raw_data/dis_time.npy')
def strftime(time):
    return time.strftime('%Y-%m-%d %H:%M:%S')

class Dispatch:
    def __init__(self, data):
        self.sender_station = data['sender_station']
        self.receiver_station = data['receiver_station']
        self.send_datetime = data['send_datetime']
        self.receive_datatime = data['receive_datetime']
        self.time = self.send_datetime
        self.time_constrain = TIME_CONSTRAINS[self.sender_station, self.receiver_station]
        self.wasting_time = 0
        self.hops = [self.sender_station]

    def reward(self):
        assert(self.hops[-1]==self.receiver_station)
        reward = -len(self.hops)+1
        if self.wasting_time < self.time_constrain:
            reward += 5

        return reward

    def deliver(self, chosed_subway):
        self.hops.append(chosed_subway['swipe_out_station'])
        self.wasting_time += (chosed_subway['swipe_out_time'] - self.time).seconds
        self.time = chosed_subway['swipe_out_time']

    def output(self):
        if self.wasting_time > self.time_constrain:
            lefttime = 0
        elif self.time+timedelta(self.time_constrain - self.wasting_time) < self.localtime:
            lefttime = 0
        else:
            lefttime = (self.time+timedelta(self.time_constrain - self.wasting_time) - self.localtime).seconds
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

    def step(self, actions):
        reward = 0
        # chose action for packages
        assert(len(actions) == len(self.dispatchs_waiting))
        waiting = []
        for i in range(len(actions)):
            if actions[i] == self.dispatchs_waiting[i].hops[-1]:
                waiting.append(self.dispatchs_waiting[i])
            else:
                avail_subway = self.subways_entering[self.subways_entering['swipe_out_station']==actions[i]]
                if not avail_subway.empty:
                    chosed_subway = avail_subway.sample().iloc[0]
                    self.dispatchs_waiting[i].deliver(chosed_subway)
                    self.dispatchs_delivering.append(self.dispatchs_waiting[i])
                    self.state['packages'][0][self.dispatchs_waiting[i].hops[-2]] -= 1
                    self.state['packages'][1][self.dispatchs_waiting[i].hops[-2], self.dispatchs_waiting[i].hops[-1]] += 1
                else:
                    waiting.append(self.dispatchs_waiting[i])
        self.dispatchs_waiting = waiting

        # set the commuting passengers
        for index, row in self.subways_entering.iterrows():
            self.state['passengers'][1][row['swipe_in_station'], row['swipe_out_station']] += 1
        self.subways_commuting = self.subways_commuting.append(self.subways_entering)

        # next episode
        self.time += self.episode
        self.step_nums += 1
        if self.step_nums == 144:
            return {
                'done': True,
                'time': self.time,
                'reward': -len(self.dispatchs_waiting)
            }

        delivering = []
        for dispatch in self.dispatchs_delivering:
            dispatch.localtime = self.time
            if dispatch.time < self.time+self.episode:
                self.state['packages'][1][dispatch.hops[-2], dispatch.hops[-1]] -= 1
                if dispatch.hops[-1] == dispatch.receiver_station:
                    reward += dispatch.reward()
                else:
                    self.dispatchs_waiting.append(dispatch)
            else:
                delivering.append(dispatch)
        
        self.dispatchs_delivering = delivering

        arrived = self.subways_commuting[self.time+self.episode>self.subways_commuting['swipe_out_time']]
        self.subways_commuting = self.subways_commuting.drop(arrived.index)
        for index, row in arrived.iterrows():
            self.state['passengers'][1][row['swipe_in_station'], row['swipe_out_station']] -= 1

        dispatchs_handling = self.dispatchs_eval[self.time+self.episode>self.dispatchs_eval['send_datetime']]
        self.dispatchs_eval = self.dispatchs_eval.drop(dispatchs_handling.index)

        for index, row in dispatchs_handling.iterrows():
            dispatch = Dispatch(row)
            self.dispatchs_waiting.append(dispatch)

        self.state['packages'][0] = torch.zeros(self.station_num)
        dispatchs_output = []
        for dispatch in self.dispatchs_waiting:
            dispatch.localtime = self.time
            dispatchs_output.append(dispatch.output())
            self.state['packages'][0][dispatch.sender_station] += 1

        self.state['dispatchs'] = dispatchs_output


        self.state['passengers'][0] = torch.zeros(self.station_num)
        self.subways_entering = self.subways_eval[self.time+self.episode>self.subways_eval['swipe_in_time']]
        self.subways_eval = self.subways_eval.drop(self.subways_entering.index)
        for index, row in self.subways_entering.iterrows():
            self.state['passengers'][0][row['swipe_in_station']] += 1


        self.state['reward'] = reward
        self.state['time'] = strftime(self.time)
        return self.state
    
    def reset(self, day=1):
        '''
        reset the environment.
        return states and time
        '''
        self.dispatchs_waiting = []
        self.dispatchs_delivering = []
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

        self.dispatchs_eval = self.dispatchs.copy(deep=True)
        self.dispatchs_eval = self.dispatchs_eval[self.dispatchs_eval['send_datetime'].apply(lambda x: x.day) == day]

        subways_eval = self.subways.copy(deep=True)
        subways_eval['swipe_in_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_in_time'])
        subways_eval['swipe_out_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_out_time'])
        subways_eval = subways_eval[subways_eval['swipe_in_time']<subways_eval['swipe_out_time']]
        subways_eval = subways_eval.sort_values(by='swipe_in_time')
        subways_eval = subways_eval.reset_index(drop=True)
        self.subways_eval = subways_eval

        self.time = datetime(2021, 11, day, 0, 0, 0)
        self.step_nums = 0
        self.state['time'] = self.time

        dispatchs_handling = self.dispatchs_eval[self.time+self.episode>self.dispatchs_eval['send_datetime']]
        self.dispatchs_eval = self.dispatchs_eval.drop(dispatchs_handling.index)
        for index, row in dispatchs_handling.iterrows():
            dispatch = Dispatch(row)
            self.dispatchs_waiting.append(dispatch)

        dispatchs_output = []
        for dispatch in self.dispatchs_waiting:
            dispatch.localtime = self.time
            dispatchs_output.append(dispatch.output())
            self.state['packages'][0][dispatch.sender_station] += 1
        self.state['dispatchs'] = dispatchs_output


        self.subways_entering = self.subways_eval[self.time+self.episode>self.subways_eval['swipe_in_time']]
        self.subways_eval = self.subways_eval.drop(self.subways_entering.index)
        for index, row in self.subways_entering.iterrows():
            self.state['passengers'][0][row['swipe_in_station']] += 1
        return self.state

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
        subways.to_csv('./dataset/subways_sorted.csv', sep=',')


if __name__ == '__main__':
    env = Myenv()
    state = env.reset()
    total_reward = 0
    while not state['done']:
        actions = []
        for i in state['dispatchs']:
            actions.append(i[1])
        state = env.step(actions)
        print(state)
        total_reward += state['reward']
    print(total_reward)
