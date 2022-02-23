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
        self.next_station = 0
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
            torch.tensor([self.left_step*10, 0])
            )).unsqueeze(0)

    def reward(self):
        assert(self.hops[-1]==self.receiver_station)
        reward = -len(self.hops)+1
        if self.left_step > -1:
            reward += 5

        return reward

    def deliver(self, chosed_subway):
        self.hops.append(chosed_subway['swipe_out_station'])
        self.arrive_step = chosed_subway['swipe_out_step']

    def step(self, action):
        pass
    
    def update(self):
        pass

    def add_states(self):
        state = torch.cat((                                     
            station_onehot(self.current_station),
            torch.tensor([self.left_step*10, 0])
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

    def step(self, actions):
        # chose action for packages
        assert(len(actions) == len(self.dispatchs_waiting))

        for i in range(len(actions)):
            self.dispatchs_waiting[i].step(actions[i])
            self.dispatchs_selected.append(self.dispatchs_waiting[i])
        
        selected = []
        for dispatch in self.dispatchs_selected:
            action = self.dispatchs_waiting[i].next_station
            avail_subway = self.subways_entering[self.subways_entering['swipe_out_station']==action]
            avail_subway = self.subways_entering[self.subways_entering['swipe_out_station']==actions[i]]
            if not avail_subway.empty:
                chosed_subway = avail_subway.sample().iloc[0]
                self.dispatchs_selected[i].deliver(chosed_subway)
                self.dispatchs_delivering.append(self.dispatchs_waiting[i])
                self.state['packages'][0][self.dispatchs_waiting[i].hops[-2]] -= 1
                self.state['packages'][1][self.dispatchs_waiting[i].hops[-2], self.dispatchs_waiting[i].hops[-1]] += 1
            else:
                waiting.append(self.dispatchs_waiting [i])
        self.dispatchs_selected = selected

        # set the commuting passengers
        for index, row in self.subways_entering.iterrows():
            self.state['passengers'][1][row['swipe_in_station'], row['swipe_out_station']] += 1

        # self.subways_commuting = self.subways_commuting.append(self.subways_entering)
        self.subways_commuting = pd.concat([self.subways_commuting, self.subways_entering])
        # next episode
        self.time += self.episode
        self.step_nums += 1

        if self.step_nums == 144:
            return None, None, True, self.time

        for dispatch in self.dispatchs_delivering:
            dispatch.step()
        for dispatch in self.dispatchs_waiting:
            dispatch.step()
        self.dispatchs_arrived = []

        delivering = []
        for dispatch in self.dispatchs_delivering:
            if dispatch.arrive_step == self.step_nums:
                self.state['packages'][1][dispatch.hops[-2], dispatch.hops[-1]] -= 1
                if dispatch.hops[-1] == dispatch.receiver_station:
                    self.dispatchs_arrived.append(dispatch)
                else:
                    self.dispatchs_waiting.append(dispatch)
            else:
                delivering.append(dispatch)
        
        self.dispatchs_delivering = delivering

        arrived = self.subways_commuting[self.subways_commuting['swipe_out_step'] == self.step_nums]
        self.subways_commuting = self.subways_commuting.drop(arrived.index)
        for index, row in arrived.iterrows():
            self.state['passengers'][1][row['swipe_in_station'], row['swipe_out_station']] -= 1

        dispatchs_handling = self.dispatchs_eval[self.dispatchs_eval['send_step']+1==self.step_nums]
        for index, row in dispatchs_handling.iterrows():
            dispatch = Dispatch(row)
            self.dispatchs_waiting.append(dispatch)

        self.state['packages'][0] = torch.zeros(self.station_num)

        dispatchs_output = []
        times = []
        for dispatch in self.dispatchs_waiting:
            source, target, lefttime = dispatch.output()
            dispatchs_output.append(torch.concat((F.one_hot(torch.tensor(source).long(), 118).unsqueeze(0), F.one_hot(torch.tensor(target).long(), 118).unsqueeze(0)), 0).unsqueeze(0))
            times.append(lefttime)
            self.state['packages'][0][dispatch.sender_station] += 1

        if dispatchs_output:
            dispatchs_output = torch.concat(dispatchs_output, 0)
            times = torch.tensor(times, dtype=torch.float).view(-1, 1)

        self.state['passengers'][0] = torch.zeros(self.station_num)
        self.subways_entering = self.subways_eval[self.subways_eval['swipe_in_step'] == self.step_nums]
        for index, row in self.subways_entering.iterrows():
            self.state['passengers'][0][row['swipe_in_station']] += 1


        self.state['reward'] = reward
        self.state['time'] = strftime(self.time)

        info = torch.zeros(2, self.station_num+1, self.station_num)
        info[0, 0] = self.state['packages'][0]
        info[0, 1:] = self.state['packages'][1]
        info[1, 0] = self.state['passengers'][0]
        info[1, 1:] = self.state['passengers'][1]
        return (dispatchs_output, times, info), self.state['reward'], self.state['done'], self.state['time']
    
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

        self.dispatchs_eval = self.dispatchs_eval.sample(n=100)
        self.dispatchs_eval = self.dispatchs_eval.sort_values(by='send_datetime')
        self.dispatchs_eval = self.dispatchs_eval.reset_index(drop=True)
        # print(self.dispatchs_eval)
        # input()

        subways_eval = self.subways.copy(deep=True)
        subways_eval['swipe_in_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_in_time'])
        subways_eval['swipe_out_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_out_time'])
        subways_eval = subways_eval[subways_eval['swipe_in_time']<subways_eval['swipe_out_time']]
        subways_eval = subways_eval.sort_values(by='swipe_in_time')
        subways_eval = subways_eval.reset_index(drop=True)
        subways_eval['swipe_in_step'] = subways_eval['swipe_in_time'].apply(self.time2step)
        subways_eval['swipe_out_step'] = subways_eval['swipe_out_time'].apply(self.time2step)
        self.subways_eval = subways_eval

        self.step_nums = 0
        self.state['time'] = self.time

        dispatchs_handling = self.dispatchs_eval[self.dispatchs_eval['send_datetime']+1 == self.step_nums]
        dispatch_waiting = []
        for index, row in dispatchs_handling.iterrows():
            dispatch = Dispatch(row)
            dispatchs_waiting.append(dispatch)
            self.state['packages'][0][dispatch.sender_station] += 1
        self.dispatchs_waiting = dispatchs_wating

        self.subways_entering = self.subways_eval[self.subways_eval['swipe_in_step'] == self.step_nums]
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
        subways.to_csv('./dataset/subways_sorted.csv', sep=',')


if __name__ == '__main__':
    env = Myenv()
    state = env.reset()
    total_reward = 0
    while True:
        dispatchs, lefttime, info = state
        if len(dispatchs) > 0:
            actions = torch.argmax(dispatchs[:, 1, :], dim=1)
        else:
            actions = torch.tensor([])
        state, reward, done, _ = env.step(actions)
        total_reward += reward
        if done:
            break
    print(total_reward)
