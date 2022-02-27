
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
day = 1
subways_handling = pd.read_csv('./dataset/subways_eval.csv', sep=',', index_col=0)

delta = torch.zeros(114, 118, 118)

for i in range(114):
    subways_handling = subways_handling[subways_handling['swipe_in_step']>=i]
    print(i)
    for o in range(118):
        for d in range(118):
            valid = subways_handling[subways_handling['swipe_in_station'] == o]
            valid = valid[valid['swipe_out_station'] == d]
            if not valid.empty:
                delta[i, o, d] = int(valid['swipe_out_step'].min() - i)
            else:
                delta[i, o, d] = 10000
torch.save(delta, './dataset/delta.pth')