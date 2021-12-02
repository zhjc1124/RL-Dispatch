
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
day = 1
subways_eval = pd.read_csv('subways_sorted.csv', sep=',', index_col=0)
subways_eval['swipe_in_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_in_time'])
subways_eval['swipe_out_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_out_time'])
subways_eval = subways_eval[subways_eval['swipe_out_time']>subways_eval['swipe_in_time']]
subways_eval = subways_eval.sort_values(by='swipe_in_time')
subways_eval = subways_eval.reset_index(drop=True)

running_time = torch.zeros(24, 118, 118, 2)
episode = timedelta(hours=1)
delta = subways_eval['swipe_out_time'] - subways_eval['swipe_in_time']
delta = delta.apply(lambda x: x.seconds)
for i in range(24):
    time = datetime(2021, 11, 1, i, 0, 0)
    subways_handling = subways_eval[time+episode>subways_eval['swipe_in_time']]
    subways_eval = subways_eval.drop(subways_handling.index)
    print(i)
    i = 10
    for o in range(118):
        for d in range(118):
            valid = subways_handling[subways_handling['swipe_in_station'] == o]
            valid = valid[valid['swipe_out_station'] == d]
            if not valid.empty:
                delta = valid['swipe_out_time'] - valid['swipe_in_time']
                delta = delta.apply(lambda x:x.seconds)
                running_time[i, o, d, 0] = delta.mean()
                running_time[i, o, d, 1] = delta.var()
torch.save(running_time, 'running_time.pth')



subways_eval = pd.read_csv('subways_sorted.csv', sep=',', index_col=0)
subways_eval['swipe_in_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_in_time'])
subways_eval['swipe_out_time'] = pd.to_datetime(f'2021-11-{day:02d} ' + subways_eval['swipe_out_time'])
subways_eval = subways_eval[subways_eval['swipe_out_time']>subways_eval['swipe_in_time']]
subways_eval = subways_eval.sort_values(by='swipe_in_time')
subways_eval = subways_eval.reset_index(drop=True)

dispatchs_eval = pd.read_csv('dispatchs_sorted.csv', sep=',', index_col=0)
dispatchs_eval['send_datetime'] = pd.to_datetime(dispatchs_eval['send_datetime'])
dispatchs_eval['receive_datetime'] = pd.to_datetime(dispatchs_eval['receive_datetime'])
waiting_time = torch.zeros(30, 24, 118, 118, 2)
episode = timedelta(hours=1)
for day in range(1, 31):
    for hour in range(24):
        time = datetime(2021, 11, day, hour, 0, 0)
        dispatchs_handling = dispatchs_eval[time+episode>dispatchs_eval['send_datetime']]
        dispatchs_eval = dispatchs_eval.drop(dispatchs_handling.index)
        print(day, hour)
        for o in range(118):
            for d in range(118):
                valid = dispatchs_handling[dispatchs_handling['sender_station'] == o]
                valid = valid[valid['receiver_station'] == d]
                if not valid.empty:
                    times = []
                    for index, item in valid.iterrows():
                        passenger = subways_eval[subways_eval['swipe_in_time']>item['send_datetime']]
                        passenger = passenger[passenger['swipe_in_station'] == o]
                        passenger = passenger[passenger['swipe_out_station'] == d]
                        if not passenger.empty:
                            times.append((passenger.iloc[0, 1] - item['send_datetime']).seconds)
                    if len(times):
                        times = np.array(times)
                        waiting_time[day, hour, o, d, 0] = times.mean()
                        waiting_time[day, hour, o, d, 1] = times.var()
torch.save(waiting_time, 'waiting_time.pth')