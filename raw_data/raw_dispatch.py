import pandas as pd
import json
import numpy as np
from datetime import datetime

# with open('./raw_data/loc_dict.json') as f:
#     loc_dict = json.load(f)

# def spherical_dist(pos1, pos2, r=6371.004):
#     pos1 = pos1 * np.pi / 180
#     pos2 = pos2 * np.pi / 180
#     cos_lat1 = np.cos(pos1[..., 1])
#     cos_lat2 = np.cos(pos2[..., 1])
#     cos_lat_d = np.cos(pos1[..., 1] - pos2[..., 1])
#     cos_lon_d = np.cos(pos1[..., 0] - pos2[..., 0])
#     return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))
    
# station_locations = np.load('./raw_data/station_locations.npy')

# dispatchs = pd.read_csv('./raw_data/RL-Dispatch_SZ_order.csv', sep=',', index_col=0)

# dispatchs['sender_station'] = 0
# dispatchs['receiver_station'] = 0
# dispatchs['send_datetime'] = 0
# dispatchs['receive_datetime'] = 0

# for index, row in dispatchs.iterrows():
#     sender = row[['sender_longitude', 'sender_latitude']].to_numpy()
#     dispatchs.loc[index, 'sender_station'] = spherical_dist(sender, station_locations).argmin()
#     receiver = row[['receiver_longitude', 'receiver_latitude']].to_numpy()
#     dispatchs.loc[index, 'receiver_station'] = spherical_dist(receiver, station_locations).argmin()
    
#     dispatchs.loc[index, 'send_datetime'] = datetime(2021, 11, dispatchs.loc[index, 'day'], dispatchs.loc[index, 'send_hour'], dispatchs.loc[index, 'send_minute'])
#     dispatchs.loc[index, 'receive_datetime'] = datetime(2021, 11, dispatchs.loc[index, 'day'], dispatchs.loc[index, 'receive_hour'], dispatchs.loc[index, 'receive_minute'])
#     if index % 100000 == 0:
#         print(index)

# dispatchs = dispatchs.drop(columns=['sender_longitude', 'sender_latitude', 'receiver_longitude', 'receiver_latitude',
#                         'day', 'send_hour', 'send_minute', 
#                             ])

# dispatchs.to_csv('./dataset/dispatchs.csv', sep=',')



dispatchs = pd.read_csv('./dataset/dispatchs.csv', sep=',', index_col=0)

# dispatchs['send_datetime'] = 0
# dispatchs['receive_datetime'] = 0

# for index, row in dispatchs.iterrows():
#     dispatchs.loc[index, 'send_datetime'] = datetime(2021, 11, dispatchs.loc[index, 'day']+1, dispatchs.loc[index, 'send_hour'], dispatchs.loc[index, 'send_minute'])
#     dispatchs.loc[index, 'receive_datetime'] = datetime(2021, 11, dispatchs.loc[index, 'day']+1, dispatchs.loc[index, 'receive_hour'], dispatchs.loc[index, 'receive_minute'])
#     if index % 100000 == 0:
#         print(index)

# dispatchs = dispatchs.drop(columns=['sender_longitude', 'sender_latitude', 'receiver_longitude', 'receiver_latitude',
#                         'day', 'send_hour', 'send_minute', 
#                             ])
# dispatchs.to_csv('./dataset/dispatchs.csv', sep=',')

