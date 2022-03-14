import pickle
import numpy as np
from pyrsistent import T
from rl_env_v2 import TIME_CONSTRAINS, Myenv, Dispatch
import matplotlib.pyplot as plt
import torch 
DELTA = torch.load('./dataset/delta.pth')
station_distances = np.load('./raw_data/station_distances.npy')

TIME_CONSTRAINS = np.zeros((118, 118))
distances_split = [-1, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, np.inf]


pro_step = np.zeros((118, 118))
pro_price = np.zeros((118, 118))

custpay = 5
hoppay = 1

for i in range(1, 9):
    dis_map = np.logical_and(station_distances>=distances_split[i-1], station_distances<distances_split[i])
    TIME_CONSTRAINS[dis_map] = i * 6
    pro_step[dis_map] = i * 1
    pro_price[dis_map] = i * 0.5
pro_price = pro_price.clip(1, 3)


with open('./analyze/PPOv2.pkl', 'rb') as f:
    PPO_env_m = pickle.load(f)

profit_rate_PPO_m = np.zeros(8)
delivery_num_PPO_m = np.zeros(8)
average_hop_PPO_m = np.zeros(8)
ctime_mean_PPO_m = np.zeros(8)

profit_rate_PPO_n = np.zeros(8)
delivery_num_PPO_n = np.zeros(8)
average_hop_PPO_n = np.zeros(8)
ctime_mean_PPO_n = np.zeros(8)

time_split_count = np.zeros(8)
for d in PPO_env_m.dispatchs_arrived:
    for i in range(0, len(d.hops)-1):
        s = d.hops[i]
        r = d.hops[i+1]
        step = d.states[i, -1]
        predict_time = DELTA[step, s, r]
        constrain = TIME_CONSTRAINS[s, r]
        if predict_time > constrain:
            time_split_class = int(TIME_CONSTRAINS/6)
            profit_rate_PPO_m[time_split_class] += custpay - i - pro_price[TIME_CONSTRAINS]
            delivery_num_PPO_m[time_split_class] += 1
            average_hop_PPO_m[time_split_class] += i + 1
            ctime_mean_PPO_m[time_split_class] += step + pro_step[TIME_CONSTRAINS] - d.send_step
            break
    else:
        time_split_class = int(TIME_CONSTRAINS/6)
        profit_rate_PPO_m[time_split_class] += custpay - (len(d.hops) - 1)
        delivery_num_PPO_m[time_split_class] += 1
        average_hop_PPO_m[time_split_class] += len(d.hops) - 1
        ctime_mean_PPO_m[time_split_class] += d.arrive_step - TIME_CONSTRAINS

    time_split_class = int(TIME_CONSTRAINS/6)
    profit_rate_PPO_n[time_split_class] += custpay - (len(d.hops) - 1)
    delivery_num_PPO_n[time_split_class] += 1
    average_hop_PPO_n[time_split_class] += len(d.hops) - 1
    ctime_mean_PPO_n[time_split_class] += d.arrive_step - TIME_CONSTRAINS

    time_split_count[time_split_class] += 1

plt.plot(np.arange(8), profit_rate_PPO_m/time_split_count)
plt.plot(np.arange(8), profit_rate_PPO_n/time_split_count)

# with open('./analyze/direct.pkl', 'rb') as f:
#     direct_env_m = pickle.load(f)


# with open('./analyze/direct.pkl', 'rb') as f:
#     direct_env = pickle.load(f)
# direct_time = np.zeros(144)
# for d in direct_env.dispatchs_arrived:
#     s = d.sender_station
#     r = d.receiver_station
#     t = d.arrive_step-d.send_step
#     if t <= TIME_CONSTRAINS[s, r]:
#         direct_time[t] += 1
# plt.plot(np.arange(144), np.cumsum(direct_time))

# with open('./analyze/PPOv2.pkl', 'rb') as f:
#     PPO_env = pickle.load(f)
# PPO_time = np.zeros(144)
# for d in PPO_env.dispatchs_arrived:
#     s = d.sender_station
#     r = d.receiver_station
#     t = d.arrive_step-d.send_step
#     if t <= TIME_CONSTRAINS[s, r]:
#         PPO_time[(d.arrive_step-d.send_step)] += 1
# plt.plot(np.arange(144), np.cumsum(PPO_time))