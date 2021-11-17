# import numpy as np
import pandas as pd
import json
import requests


def station_GPS(station):
    url = 'https://restapi.amap.com/v3/geocode/geo'
    params = {
        'key': '71e04b5bbf5f450e1987b06e58e6e33a',
        'address': station+'地铁站',
        'city': '深圳',
    }
    res = requests.get(url, params=params)
    location = res.json()['geocodes'][0]['location']
    lat, lon = location.split(',')
    lat = float(lat)
    lon = float(lon)
    return lat, lon


SmartCardData = pd.read_csv('SmartCardData.csv', sep=',', header=None)
SmartCardData = SmartCardData[SmartCardData[2] != 31]
SmartCardData.sort_values(by=[0, 1, 2], inplace=True)
SmartCardData = SmartCardData.reset_index(drop=True)

counts = SmartCardData[3].value_counts()
# station_locations = np.zeros((len(counts)-1, 2))
loc_dict = {}
for num, c in enumerate(counts[1:].index):
    loc_dict[c] = num
    # station_locations[num] = station_GPS(c)
with open('loc_dict.json', 'w') as f:
    json.dump(loc_dict, f)

# np.save('station_locations.npy', station_locations)

subways_columns = ['card_id', 'swipe_in_time', 'swipe_in_station',
                              'swipe_out_time', 'swipe_out_station']
subways = pd.DataFrame(columns=subways_columns)
columns_dtypes = [str, str, int, str, int]
subways = subways.astype(dict(zip(subways_columns, columns_dtypes)))

s = [0]*5
line = 0
for i in range(SmartCardData.shape[0]):
    record = SmartCardData.loc[i]
    if s[0] == 0:
        s[0] = record[0]
        if record[2] == 21:
            s[1] = record[1]
            s[2] = record[3]
        elif record[2] == 22:
            s[3] = record[1]
            s[4] = record[3]
    elif s[0] == record[0]:
        if record[2] == 21:
            s[1] = record[1]
            s[2] = record[3]
        elif record[2] == 22:
            s[3] = record[1]
            s[4] = record[3]
        if s[2] == 'None' or s[4] == 'None':
            s = [0] * 5
            continue
        if s[2] == 0 or s[4] == 0:
            s = [0] * 5
            continue
        s[2] = loc_dict[s[2]]
        s[4] = loc_dict[s[4]]
        fw = open('../dataset/subways.csv', 'a+')
        fw.write(str(line)+',')
        for index, k in enumerate(s):
            fw.write(str(k))
            if index == len(s)-1:
                fw.write('\n')
            else:
                fw.write(',')
        s = [0] * 5
        line += 1
    else:
        s = [0] * 5
        continue
    if i % 1000 == 0:
        print(i)

subways.to_csv('../dataset/subways.csv')