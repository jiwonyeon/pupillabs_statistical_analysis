#%% import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os, glob
import subprocess, pickle
import plotly.graph_objects as go 
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


#%% set directories
dataPath = os.path.abspath(glob.glob('./data/2023*')[0])
figPath= './figure'

# if the pkl file does not exist, generate the pkl file first
pupil_to_pkl = os.path.abspath('../pupillabs_util/pupil_to_pkl.py')
command = ["python", pupil_to_pkl, dataPath]
if not os.path.exists(os.path.join(dataPath, 'eyedata.pkl')):
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

# load pupil_data
pupil_data = pickle.load(open(os.path.join(dataPath, 'eyedata.pkl'), 'rb'))


#%% figure generation
# select the data to plot
chopping = np.arange(0,3001)
times = (pupil_data['gaze']['timestamp [ns]'] - pupil_data['gaze']['timestamp [ns]'][0]) / 1e9
times = times[chopping]
times.name = 'time [sec]'

# extract gazes and fixation start and end points
gaze = pupil_data['gaze'].iloc[chopping]
fixation_start = []
fixation_end = []
for id in gaze['fixation id'].dropna().unique():
    fixation = gaze[gaze['fixation id']==id]
    fixation_start.append(fixation.index[0])
    fixation_end.append(fixation.index[-1])    

# compute velocity of x and y direction
fsp = np.average(np.diff(times))
velocity_x = np.gradient(gaze['azimuth [deg]'], fsp)
velocity_y = np.gradient(gaze['elevation [deg]'], fsp)

# compute velocity considering both directions
velocity_magnitude = np.sqrt(velocity_x**2+velocity_y**2)

# second derivative of the amplitude
acceleration = np.gradient(velocity_magnitude, fsp)

# first, threshold the acceleration 
threshold = 35/fsp      # threshold for acceleration value
acceleration_thresholded = np.where(acceleration>=threshold)[0]

# after thresholding, find the peak acceleration within the 100 ms window
peak_acc_idx = []
window = 0.15   # 150 ms
for i in acceleration_thresholded:
    this_window = np.where((times[i]-window/2 < times) & (times < times[i]+window/2))[0]
    peak_acc_idx.append(np.argmax(acceleration[this_window])+this_window[0])
peak_acc_idx = np.unique(peak_acc_idx)   # remain only unique values

# refine the peak acceleration 
for i in peak_acc_idx:
    this_window = np.where((times[i]-window/2 < times) & (times < times[i]+window/2))[0]
    acc = acceleration[this_window]
    if np.argmax(acc)!=(i-this_window[0]):
        peak_acc_idx = peak_acc_idx[peak_acc_idx != i]

# find saccade durations based on the peak acceleration index
saccade_start = []
saccade_end = []
for peak_id, i in enumerate(peak_acc_idx):
    # find the start of saccade duration
    this_window = np.where((times >= times[i]-0.035) & (times <= times[i]))[0]  # between -35 ms to the peak
    acc_diff = np.diff(acceleration[this_window])
    saccade_start.append(np.argmin(np.abs(acc_diff))+this_window[0]+1)   # find the derivative closest to 0 

    # find the end of saccade duration
    # first find the deep that happens until the next peak
    if i < peak_acc_idx[-1]:
        this_window = np.arange(i, peak_acc_idx[peak_id+1])
    else:
        this_window = np.arange(i,len(times))
    deep = np.argmin(acceleration[this_window])+this_window[0]
    
    # find the first most flat point after 100ms of the deep
    this_window = np.where((times > times[deep]) & (times < times[deep]+0.1))[0]
    acc_diff = np.diff(acceleration[deep:this_window[-1]])
    saccade_end.append(np.argmin(np.abs(acc_diff))+deep+1)

# find velocity peaks, around the acceleration peak
peak_velocity_idx = []
window = 0.15   # 150 ms
for i in peak_acc_idx:
    this_window = np.where((times[i]-window/2 < times) & (times < times[i]+window/2))[0]
    peak_velocity_idx.append(np.argmax(velocity_magnitude[this_window])+this_window[0])
peak_velocity_idx = np.unique(peak_velocity_idx)   # remain only unique values

# refine the velocity_peak
for i in peak_velocity_idx:
    this_window = np.where((times[i]-window/2 < times) & (times < times[i]+window/2))[0]
    v = velocity_magnitude[this_window]
    if np.argmax(v)!=(i-this_window[0]):
        peak_velocity_idx = peak_velocity_idx[peak_velocity_idx != i]

# create figure for plotting the acceleration, velocity, and the raw data
fig, ax = plt.subplots(figsize=(10,10), nrows=4, ncols=1)

ax[0].plot(times, acceleration, alpha=0.5, label='acceleration')
ax[0].scatter(times[peak_acc_idx], acceleration[peak_acc_idx], 
            color='red', label='peak acceleration')
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax[0].axvspan(start, end, color='gray', alpha=.2)
ax[0].set_ylabel(r'Acceleration[deg/sec$^{-2}$]')
ax[0].legend()

ax[1].plot(times, velocity_x, alpha=0.5, label='azimuth')
ax[1].plot(times, velocity_y, alpha=0.5, label='elevation')
ax[1].plot(times, velocity_magnitude, alpha=0.8, label='magnitude')
ax[1].scatter(times[peak_velocity_idx], velocity_magnitude[peak_velocity_idx], color='red', label='peak velocity')
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax[1].axvspan(start, end, color='gray', alpha=.2)
ax[1].set_ylabel('Velocity [deg/s]')
ax[1].legend(loc='lower right')

ax[2].plot(times, gaze['azimuth [deg]'])
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax[2].axvspan(start, end, color='gray', alpha=.2)
ax[2].set_ylabel('Azimuth [deg]')

ax[3].plot(times, gaze['elevation [deg]'])
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax[3].axvspan(start, end, color='gray', alpha=.2)
ax[3].set_xlabel('Time [sec]')
ax[3].set_ylabel('Elevation [deg]')

