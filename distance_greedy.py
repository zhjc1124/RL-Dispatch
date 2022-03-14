import numpy as np
import torch
from rl_env_v2 import Myenv, UPPERBOUND

station_location = np.load('./raw_data/station_locations.npy')

station_distances = np.load('./raw_data/station_distances.npy')

def distance_greedy(env, dispatch_waiting):
    current_station = dispatch_waiting.hops[-1]                # 当前站点
    receiver_station = dispatch_waiting.receiver_station   # 接收站点
    subways_entering = env.subways_entering
    valid_stations = subways_entering[subways_entering['swipe_in_station'] == current_station]
    if len(valid_stations):
        valid_stations = np.array(list(set(valid_stations['swipe_out_station']))).astype(np.int32)
        current_distance = station_distances[current_station, receiver_station]
        valid_distances = station_distances[current_station, valid_stations]
        valid_distances[current_distance < valid_distances] = np.inf
        if np.isinf(valid_distances).all():
            return current_station
        else:
            return valid_stations[valid_distances.argmin()]
    else:
        return current_station

    # pp = env.state['passengers'][1]                                  # 乘客分布
    # index_pp = torch.arange(pp.shape[0])
    # pp_index = index_pp[pp[dispatchs_station_now, :] != 0]              # 可选择乘客目的点
    # # pp_now = pp[dispatchs_station_now, pp[dispatchs_station_now, :] != 0]                   # 可选择乘客
    # # station_distance[dispatchs_station_receiver, pp_index].argmin()
    # if min(pp_index.shape) == 0:
    #     action = dispatchs_station_receiver
    # else:
    #     action = pp_index[station_distances[dispatchs_station_receiver, pp_index].argmin()]
    # return action


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
            print(env.total_profit())
            print(len(env.dispatchs_arrived))
            print(env.evaluate())
            import pickle
            with open('./analyze/distance_greedy.pkl', 'wb') as f:
                pickle.dump(env, f)
            break