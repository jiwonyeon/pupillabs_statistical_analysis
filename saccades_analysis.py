#%% import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os, glob
import subprocess, pickle
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

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


#%% detect saccades and generate figures
# select the data to plot
chopping = np.arange(0,3501)
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
fsp = round(np.average(np.diff(times)),3)
velocity_x = np.gradient(gaze['azimuth [deg]'], fsp) 
velocity_y = np.gradient(gaze['elevation [deg]'], fsp)

# compute velocity considering both directions
velocity_magnitude = np.sqrt(velocity_x**2+velocity_y**2)

# second derivative of the amplitude
acceleration = np.gradient(velocity_magnitude, fsp)

# threshold the velocity_magnitude
threshold = 150      # threshold for velocity_magnitude
velocity_thresholded = np.where(velocity_magnitude>=threshold)[0]

# apply butterworth filter
cutoff = 18     # cutoff frequency
order = 2   # filter order
nyq = 0.5*(1/fsp)  # Nyquist frequency, frequency in Hz
normal_cutoff = cutoff/nyq

# filter coefficients
b, a = butter(order, normal_cutoff, btype='low', analog=False)
filtered_acceleration = filtfilt(b,a,acceleration)

# find the peak acceleration within a window
peak_acc_idx = []
window = 0.15   # 150 ms
for i in velocity_thresholded:
    this_window = np.where((times[i]-window/2 < times) & (times < times[i]+window/2))[0]
    peak_acc_idx.append(np.argmax(acceleration[this_window])+this_window[0])
peak_acc_idx = np.unique(peak_acc_idx)      # reserve only unique values

# refine the peak acceleration 
for i in peak_acc_idx:
    this_window = np.where((times[i]-window/2 < times) & (times < times[i]+window/2))[0]
    acc = acceleration[this_window]

    # if the time of detected max value of this saccade does not match to the index, throw out the index
    if np.argmax(acc)!=(i-this_window[0]):      
        peak_acc_idx = peak_acc_idx[peak_acc_idx != i]

# find saccade durations based on the filtered_acceelration
window = 0.2       # 200ms 
buffer = 0.01     # 25ms, buffer for the start and end of the saccade 
saccade_start = []
saccade_end = []
for peak_id, t_id in enumerate(peak_acc_idx):
    # find the start of a saccade 
    this_window = np.where((times >= times[t_id]-window/2)& (times <= times[t_id]+window/2))[0]
    this_saccade = filtered_acceleration[this_window]

    # find saccade start: start of the dip before the peak of the saccade  
    pre_saccade_dip = np.argmin(filtered_acceleration[this_window[0]:t_id])+this_window[0]
    saccade_start_idx = np.argmin(np.abs(times - (times[pre_saccade_dip]-buffer)))
    saccade_start.append(saccade_start_idx)

    # find saccade end: endn of the bump after the dip of the saccade
    saccade_dip = np.argmin(filtered_acceleration[t_id:this_window[-1]])+t_id
    post_saccade_bump = np.argmax(filtered_acceleration[saccade_dip:this_window[-1]])+saccade_dip
    saccade_end_idx = np.argmin(np.abs(times - (times[post_saccade_bump]+buffer)))
    saccade_end.append(saccade_end_idx)
    
# TODO: need to compute amplitude of the saccades 
# TODO: What is the unit of deg/fsp and deg/fsp^2 and how do I convert it to deg/s?

# Figure 1. acceleration
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
ax1.plot(times, acceleration, label = 'Acceleration')
ax1.plot(times, filtered_acceleration, label = 'Fitlered acceleration')
ax1.scatter(times[peak_acc_idx], acceleration[peak_acc_idx], color='red', label='Peak')
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax1.axvspan(start, end, color='gray', alpha=.2)
ax1.set_ylabel(r'Acceleration[deg/sec$^{-2}$]')
ax1.legend()

# Figure 2. Velocity
ax2.plot(times, velocity_magnitude, label = 'magnitude')


# # create figure for plotting the acceleration, velocity, and the raw data
# fig, ax = plt.subplots(figsize=(10,10), nrows=4, ncols=1)

# ax[0].plot(times, acceleration, alpha=0.5, label='acceleration')
# ax[0].scatter(times[peak_acc_idx], acceleration[peak_acc_idx], 
#             color='red', label='peak acceleration')
# for start, end in zip(times[saccade_start], times[saccade_end]):
#     ax[0].axvspan(start, end, color='gray', alpha=.2)
# ax[0].set_ylabel
# ax[0].legend()

# ax[1].plot(times, velocity_x, alpha=0.5, label='azimuth')
# ax[1].plot(times, velocity_y, alpha=0.5, label='elevation')
# ax[1].plot(times, velocity_magnitude, alpha=0.8, label='magnitude')
# ax[1].scatter(times[peak_velocity_idx], velocity_magnitude[peak_velocity_idx], color='red', label='peak velocity')
# for start, end in zip(times[saccade_start], times[saccade_end]):
#     ax[1].axvspan(start, end, color='gray', alpha=.2)
# ax[1].set_ylabel('Velocity [deg/s]')
# ax[1].legend(loc='lower right')

# ax[2].plot(times, gaze['azimuth [deg]'])
# for start, end in zip(times[saccade_start], times[saccade_end]):
#     ax[2].axvspan(start, end, color='gray', alpha=.2)
# ax[2].set_ylabel('Azimuth [deg]')

# ax[3].plot(times, gaze['elevation [deg]'])
# for start, end in zip(times[saccade_start], times[saccade_end]):
#     ax[3].axvspan(start, end, color='gray', alpha=.2)
# ax[3].set_xlabel('Time [sec]')
# ax[3].set_ylabel('Elevation [deg]')

# #%% compare saccade duration with amplitude and maximum acceleration
# saccade_duration = times[saccade_end].to_numpy()-times[saccade_start].to_numpy()
# saccade_max_acceleration = acceleration[peak_acc_idx]
# saccade_amplitude = acceleration[peak_acc_idx]-acceleration[neg_peak_acc_idx]

# order = np.argsort(saccade_duration)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
# n = 0
# ax[n].plot(saccade_duration[order], saccade_amplitude, '-o')
# ax[n].set_xlabel('Saccade duration [sec]')
# ax[n].set_ylabel('Saccade amplitude')

# n += 1
# ax[n].plot(saccade_duration[order], saccade_max_acceleration[order], '-o')
# ax[n].set_xlabel('Saccade duration [sec]')
# ax[n].set_ylabel(r'Max acceleration [deg/sec$^{-2}$]')
