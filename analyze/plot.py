import pickle
import numpy as np
from ..rl_env_v2 import Myenv
with open('./analyze/direct.pkl', 'rb') as f:
    direct_env = pickle.load(f)
with open('./analyze/PPOv2.pkl', 'rb') as f:
    PPO_env = pickle.load(f)


direct_cdf = np.zeros(144)
for step in range(144):
    for d in direct_env.arrived:
        print(d)

