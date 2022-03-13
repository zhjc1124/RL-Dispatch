# -------------------------------------------------
# Description:
# Reference:
# Author:   Wang Shengpeng
# encoding: utf-8
# Date:     2022/3/13

import numpy as np

import torch

from rl_env_v2 import Myenv, UPPERBOUND

station_distance = np.load('./raw_data/dis_time.npy')

def distance_greedy(env, dispatch_waiting):
    dispatchs_station_now = dispatch_waiting.hops[-1]                # 当前站点
    dispatchs_station_receiver = dispatch_waiting.receiver_station   # 接收站点
    pp = env.state['passengers'][1]                                  # 乘客分布
    index_pp = torch.arange(pp.shape[0])
    pp_index = index_pp[pp[dispatchs_station_now, :] != 0]              # 可选择乘客目的点
    # pp_now = pp[dispatchs_station_now, pp[dispatchs_station_now, :] != 0]                   # 可选择乘客
    # station_distance[dispatchs_station_receiver, pp_index].argmin()
    if min(pp_index.shape) == 0:
        action = dispatchs_station_receiver
    else:
        action = pp_index[station_distance[dispatchs_station_receiver, pp_index].argmin()]
    return action


if __name__ == '__main__':
    env = Myenv()
    dispatchs, done, _ = env.reset()
    while True:
        actions = []
        for dispatch in dispatchs:
            action = distance_greedy(env, dispatch)
            actions.append(action)
        dispatchs, done, _ = env.step(actions)
        if done:
            print(env.total_reward())
            print(len(env.dispatchs_arrived))
            print(env.evaluate())
            break