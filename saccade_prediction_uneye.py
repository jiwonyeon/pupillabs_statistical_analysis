### saccade detection test with uneye
### the prediction is done with a pretrained network by uneye
### jiwon yeon, march 2024

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob, pickle, os, sys
import scipy.io as io
import uneye

sys.path.append('Users/jyeon/Documents/GitHub/uneye')

# load data
datapath = './data/2023-10-13_12-52-19-b960757a'
eyedata = pickle.load(open(os.path.join(datapath, 'eyedata.pkl'), 'rb'))
times = eyedata['gaze']['timestamp [ns]'].values
times = (times - times[0]) / 1e9

# params // most params follow the tutorial
fsp = int(1/round(np.average(np.diff(times)),3))     # sampling rate in Hz
weights_name = 'weights_Andersson'
min_sacc_dur = 6    # in ms
min_sacc_dist = 10  # in ms

# x and y positions of gaze. all data will be used for training.
x = eyedata['gaze']['azimuth [deg]'].values
y = eyedata['gaze']['elevation [deg]'].values

# prediction model
uneye_model = uneye.DNN(weights_name=weights_name, 
                        sampfreq=fsp, 
                        min_sacc_dur=min_sacc_dur,
                        min_sacc_dist=min_sacc_dist)
Prediction, Probability = uneye_model.predict(x, y)

# plot example 
plot_time = np.arange(3000,8000)
fig = plt.figure(figsize=(10,5))
plt.plot(times[plot_time], x[plot_time]-np.min(x[plot_time]))
plt.plot(times[plot_time], y[plot_time]-np.min(y[plot_time]))
plt.ylabel('Relative eye position [deg]')
plt.xlabel('Time [sec]')

for i in plot_time:
    if Prediction[i] == 1:
        plt.fill_between(times, Prediction*100, 
                         where=(times >= times[i])&(times < times[i+1]),
                         color='gray', alpha=0.5)
        

plt.show()

