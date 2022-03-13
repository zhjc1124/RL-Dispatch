# import numpy as np
import pandas as pd
import json
import requests
import numpy as np
import time

def spherical_dist(pos1, pos2, r=6371.004):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 1])
    cos_lat2 = np.cos(pos2[..., 1])
    cos_lat_d = np.cos(pos1[..., 1] - pos2[..., 1])
    cos_lon_d = np.cos(pos1[..., 0] - pos2[..., 0])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

def gaode_GPS(station):
    url = 'https://restapi.amap.com/v3/place/text'
    params = {
        'key': '71e04b5bbf5f450e1987b06e58e6e33a',
        'keywords': station + '站',
        'types': 150500,
        'city': 440300,
        'citylimit': 'true',
        'offset': 1,
    }
    res = requests.get(url, params=params)
    location = res.json()['pois'][0]['location']
    lon, lat = location.split(',')
    lon = float(lon)
    lat = float(lat)
    return lon, lat

def baidu_GPS(station):
    url = 'https://api.map.baidu.com/place/v2/search'
    params = {
        'ak': 'UrfIaT7weuZH3PI4imft9IZ8r0KRWZWi',
        'query': station + '站',
        'tag': '地铁站',
        'city_limit': 'true',
        # 'page_size': 1,
        'region': '深圳市',
        'output': 'json'
    }
    res = requests.get(url, params=params)
    location = res.json()['results'][0]['location']
    lon = float(location['lng'])
    lat = float(location['lat'])
    return lon, lat

# SmartCardData = pd.read_csv('SmartCardData.csv', sep=',', header=None)
# SmartCardData = SmartCardData[SmartCardData[2] != 31]
# SmartCardData.sort_values(by=[0, 1, 2], inplace=True)
# SmartCardData = SmartCardData.reset_index(drop=True)

# counts = SmartCardData[3].value_counts()
# station_locations = np.zeros((len(counts)-1, 2))
# loc_dict = {}
# for num, c in enumerate(counts[1:].index):
#     loc_dict[c] = num
#     # station_locations[num] = station_GPS(c)
# with open('loc_dict.json', 'w') as f:
#     json.dump(loc_dict, f)

station_locations = np.zeros((118, 2))
station_distances = np.zeros((118, 118))
with open('./raw_data/loc_dict.json', 'r') as f:
    loc_dict = json.load(f)
for key in loc_dict:
    num = loc_dict[key]
    c = key
    station_locations[num] = gaode_GPS(c)
    time.sleep(0.1)
    print(station_locations[num], c, num)
    pass

for i in range(118):
    for j in range(i, 118):
        if i == j:
            station_distances[i, j] = np.inf
        else:
            station_distances[i, j] = station_distances[j, i] = spherical_dist(station_locations[i], station_locations[j])
            
np.save('./raw_data/station_locations.npy', station_locations)
np.save('./raw_data/station_distances.npy', station_distances)

# subways_columns = ['card_id', 'swipe_in_time', 'swipe_in_station',
#                               'swipe_out_time', 'swipe_out_station']
# subways = pd.DataFrame(columns=subways_columns)
# columns_dtypes = [str, str, int, str, int]
# subways = subways.astype(dict(zip(subways_columns, columns_dtypes)))

# s = [0]*5
# line = 0
# for i in range(SmartCardData.shape[0]):
#     record = SmartCardData.loc[i]
#     if s[0] == 0:
#         s[0] = record[0]
#         if record[2] == 21:
#             s[1] = record[1]
#             s[2] = record[3]
#         elif record[2] == 22:
#             s[3] = record[1]
#             s[4] = record[3]
#     elif s[0] == record[0]:
#         if record[2] == 21:
#             s[1] = record[1]
#             s[2] = record[3]
#         elif record[2] == 22:
#             s[3] = record[1]
#             s[4] = record[3]
#         if s[2] == 'None' or s[4] == 'None':
#             s = [0] * 5
#             continue
#         if s[2] == 0 or s[4] == 0:
#             s = [0] * 5
#             continue
#         s[2] = loc_dict[s[2]]
#         s[4] = loc_dict[s[4]]
#         fw = open('../dataset/subways.csv', 'a+')
#         fw.write(str(line)+',')
#         for index, k in enumerate(s):
#             fw.write(str(k))
#             if index == len(s)-1:
#                 fw.write('\n')
#             else:
#                 fw.write(',')
#         s = [0] * 5
#         line += 1
#     else:
#         s = [0] * 5
#         continue
#     if i % 1000 == 0:
#         print(i)

# subways.to_csv('../dataset/subways.csv')