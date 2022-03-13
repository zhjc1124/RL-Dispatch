import numpy as np
import matplotlib.pyplot as plt

time_level = 8
station_distances = np.load('./raw_data/station_distances.npy')

dis_time = np.zeros((118, 118))

dis = station_distances.flatten()
dis = dis[~np.isinf(dis)]
min_distance = dis.min()
max_distance = dis.max()
step_distance = (max_distance-min_distance)/time_level

dis_time[np.isinf(station_distances)] = np.inf
for i in range(time_level):
    left = min_distance + i*step_distance
    right = min_distance + (i+1)*step_distance
    dis_time[np.logical_and(left <= station_distances, station_distances <= right)] = i+1

np.save('./raw_data/dis_time.npy', station_distances)