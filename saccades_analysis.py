#%% import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os, glob
import subprocess, pickle
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit

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
chopping = np.arange(0,50000)
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
buffer = 0.02     # buffer for the start and end of the saccade 
saccade_start = []
saccade_end = []
for peak_id, peak_time in enumerate(peak_acc_idx):
    if peak_id != len(peak_acc_idx)-1:
        next_peak_time = peak_acc_idx[peak_id+1]
    else:
        next_peak_time = len(times)-1
    window_start = np.argmin(np.abs(times - (times[peak_time]-0.1)))

    # if is a noisy peak-time, adjust it
    while filtered_acceleration[peak_time] < np.max(filtered_acceleration[window_start:peak_time]):
        peak_time = np.argmax(filtered_acceleration[window_start:peak_time])+window_start
        window_start = np.argmin(np.abs(times - (times[peak_time]-0.1)))

    # find the dip in the saccade
    pre_saccade_dip = np.argmin(filtered_acceleration[window_start:peak_time])+window_start

    # set saccade start time
    saccade_start_idx = np.argmin(np.abs(times - (times[pre_saccade_dip]-buffer)))
    
    # make sure not to overlap with the previous saccade
    if (len(saccade_end) != 0):
        if (saccade_start_idx < saccade_end[-1]):
            saccade_start_idx = saccade_end[-1]+1
    saccade_start.append(saccade_start_idx)

    # find the dip in the saccade
    post_saccade_dip = np.argmin(filtered_acceleration[peak_time:next_peak_time])+peak_time






    

    # if the current and the next peak are not too close, find bump after a saccade
    if (np.abs(times[peak_time]-times[next_peak_time]) > 0.2):
        if peak_id != len(peak_acc_idx)-1:    
            window_end = np.argmin(np.abs(times - (times[next_peak_time]-0.05)))
        else:
            window_end = len(times)-1       # case where it's the last 

        # find where the oscillation gets converged
        test = []
        for i, idx in enumerate(np.arange(post_saccade_dip, window_end)):
            test.append(max(filtered_acceleration[idx:idx+20])-min(filtered_acceleration[idx:idx+20]))
            
        # apply butterworth filter
        cutoff = 10     # cutoff frequency
        order = 2   # filter order
        nyq = 0.5*(1/fsp)  # Nyquist frequency, frequency in Hz
        normal_cutoff = cutoff/nyq

        # filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        test2 = filtfilt(b,a,filtered_acceleration)

        test3 = []
        for i in np.arange(saccade_start[3], saccade_end[3]):
            std = np.std(filtered_acceleration[i:i+20])
            test3.append(std)

        bump = np.argmax(filtered_acceleration[post_saccade_dip:window_end])+post_saccade_dip
        saccade_end_idx = np.argmin(np.abs(times - (times[bump]+buffer)))

    else:   # if the current and the next peak are too close, find a middle ground
        middle_point = int(np.average([post_saccade_dip, next_peak_time]))
        if post_saccade_dip < middle_point:
            saccade_end_idx = np.argmin(np.abs(filtered_acceleration[post_saccade_dip:middle_point]))+post_saccade_dip
        else:
            saccade_end_idx = post_saccade_dip

    saccade_end.append(saccade_end_idx)


   
# compute duration, amplitude, and peak velocity of saccade
combined_amplitude = np.sqrt(gaze['azimuth [deg]']**2 + gaze['elevation [deg]']**2)
peak_velocity = []
amplitude = []
for start, end in zip(saccade_start, saccade_end):
    # compute amplitude of saccade
    amplitude.append(np.max(combined_amplitude[start:end])-np.min(combined_amplitude[start:end]))

    # compute peak velocity 
    peak_velocity.append(np.max(velocity_magnitude[start:end]))    
amplitude = np.array(amplitude)
peak_velocity = np.array(peak_velocity)
saccade_duration = times[saccade_end].to_numpy()-times[saccade_start].to_numpy()


#%% Figure 1. plot raw data
fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
ax1.plot(times, acceleration, label = 'acceleration')
ax1.plot(times, filtered_acceleration, label = 'filtered_acceleration')
ax1.scatter(times[peak_acc_idx], acceleration[peak_acc_idx], color='red', label='Peak')
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax1.axvspan(start, end, color='gray', alpha=.2)
ax1.set_ylabel(r'Acceleration[deg/sec$^{-2}$]')
ax1.legend()

# ax2. velocity magnitude
ax2.plot(times, velocity_magnitude, label = 'magnitude')
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax2.axvspan(start, end, color='gray', alpha=.2)
ax2.set_ylabel(r'Velocity[deg/sec]')
ax2.legend()

# ax3. velocity for azimuth and elevation
ax3.plot(times, velocity_x, label = 'azimuth')
ax3.plot(times, velocity_y, label = 'elevation')
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax3.axvspan(start, end, color='gray', alpha=.2)
ax3.set_ylabel(r'Velocity[deg/sec]')
ax3.legend()
ax3.set_xlabel('Time [sec]')

#%% Figure 2. plot saccade amplitude and velocity against duration
sort_order = np.argsort(saccade_duration)
fig2, (ax1,ax2) = plt.subplots(2,1)
ax1.scatter(saccade_duration[sort_order], amplitude[sort_order])
ax1.set_ylabel('Amplitude [deg]')
ax1.set_xticklabels([])

ax2.scatter(saccade_duration[sort_order], peak_velocity[sort_order])
ax2.set_ylabel('Peak Velocity [deg/sec]')
ax2.set_xlabel('Time [sec]')



plt.figure()
alpha = np.linspace(0.1, 1.0, len(thiswindow))
plt.scatter(gaze['azimuth [deg]'].loc[thiswindow], gaze['elevation [deg]'].loc[thiswindow], color='blue', alpha=alpha)