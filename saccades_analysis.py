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
# dataPath = os.path.abspath(glob.glob('./data/2023*')[1])
dataPath = os.path.abspath('./data/2023-10-17_14-10-50-482637ef')
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
chopping = np.arange(0,30000)
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
time_diff = np.append(np.diff(times), np.diff(times)[-1])

azimuth = gaze['azimuth [deg]']
elevation = gaze['elevation [deg]']
amplitude = np.sqrt(azimuth**2 + elevation**2)
velocity = np.gradient(amplitude, fsp)
# velocity_2 = np.gradient(amplitude, time_diff)

plt.figure()
plt.plot(times[:5000], azimuth[:5000], label='azimuth')
plt.plot(times[:5000], elevation[:5000], label='elevation')
plt.plot(times[:5000], amplitude[:5000], label='amplitude')
plt.title('Pupil - Lego / ms')
plt.legend(loc='lower right')
plt.xlabel('Time [sec]')
plt.ylabel('Angle [deg]')

plt.figure()
plt.plot(time_diff[:1000])
plt.title('Pupil Labs')
plt.xlabel('Data point')
plt.ylabel('Time interval between the two data points [sec]')

plt.figure()
plt.plot(times[:5000], velocity[:5000], label='velocity')
plt.title('Pupil - Lego / ms')
plt.xlabel('Time [sec]')
plt.ylabel('Velocity [deg/sec]')


# compute velocity considering both directions
# velocity_magnitude = np.sqrt(velocity_x**2+velocity_y**2)

# second derivative of the amplitude
acceleration = np.gradient(velocity_magnitude, fsp)

# threshold the velocity_magnitude
threshold = 200      # threshold for velocity_magnitude
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

# apply butterworth filter to filtered_acceleration to smooth the signal
cutoff = 10     # cutoff frequency
order = 2   # filter order
nyq = 0.5*(1/fsp)  # Nyquist frequency, frequency in Hz
normal_cutoff = cutoff/nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)
saccade_tail = filtfilt(b,a,acceleration)

for peak_id, peak_time in enumerate(peak_acc_idx):
    if peak_id != len(peak_acc_idx)-1:
        next_peak_time = peak_acc_idx[peak_id+1]
    else:
        next_peak_time = len(times)-1

    window_start = np.argmin(np.abs(times - (times[peak_time]-0.1)))
    if peak_id != 0:
        if window_start < saccade_end[-1]:
            window_start = saccade_end[-1]
    
    # find the dip in the saccade
    pre_saccade_dip = np.argmin(filtered_acceleration[window_start:peak_time])+window_start

    # set saccade start time
    saccade_start_idx = np.argmin(np.abs(times - (times[pre_saccade_dip]-buffer)))
    
    # make sure not to overlap with the previous saccade
    if (len(saccade_end) != 0):
        if (saccade_start_idx < saccade_end[-1]):
            saccade_start_idx = saccade_end[-1]+1
    saccade_start.append(saccade_start_idx)

    # set a rough end point for the saccade 
    window_end = int(np.average([peak_time, next_peak_time]))
    if peak_id == len(peak_acc_idx)-1:
        window_end = len(times)-1

    # find the dip in the saccade
    post_saccade_dip = np.argmin(filtered_acceleration[peak_time:window_end])+peak_time

    # find the first point where the smoothed acceleration converges
    saccade_tail_threshold = 200
    sup_threshold = np.where(np.abs(saccade_tail[post_saccade_dip:window_end])<saccade_tail_threshold)[0]
    while len(sup_threshold) == 0:
        saccade_tail_threshold += 100
        sup_threshold = np.where(np.abs(saccade_tail[post_saccade_dip:window_end])<saccade_tail_threshold)[0]
    saccade_tail_start = sup_threshold[0] + post_saccade_dip
    
    # add a buffer to the end of the saccade
    saccade_end_idx = np.argmin(np.abs(times - (times[saccade_tail_start]+buffer)))

    # add the saccade start and end to the list
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
ax2.scatter(times[peak_acc_idx], velocity_magnitude[peak_acc_idx], color='red', label='saccade peak')
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax2.axvspan(start, end, color='gray', alpha=.2)
ax2.set_ylabel(r'Velocity[deg/sec]')

# ax3. azimuth and elevation
ax3.plot(times, gaze['azimuth [deg]'], label = 'azimuth')
ax3.plot(times, gaze['elevation [deg]'], label = 'elevation')
ax3.plot(times, combined_amplitude, linestyle='-', label = 'combined')
for start, end in zip(times[saccade_start], times[saccade_end]):
    ax3.axvspan(start, end, color='gray', alpha=.2)
ax3.set_ylabel(r'amplitude[deg]')
ax3.legend()
ax3.set_xlabel('Time [sec]')

#%% Figure 2. plot saccade amplitude and velocity against duration
sort_order = np.argsort(saccade_duration)
fig2, (ax1,ax2,ax3) = plt.subplots(3,1)
ax1.scatter(saccade_duration[sort_order], amplitude[sort_order])
ax1.set_ylabel('Amplitude [deg]')
ax1.set_xlabel('Saccade Duration [sec]')
# ax1.set_xticklabels([])

ax2.scatter(saccade_duration[sort_order], peak_velocity[sort_order])
ax2.set_ylabel('Peak Velocity [deg/sec]')
ax2.set_xlabel('Saccade Duration [sec]')

sort_order = np.argsort(amplitude)
ax3.scatter(amplitude[sort_order], peak_velocity[sort_order])
ax3.set_ylabel('Peak Velocity [deg/sec]')
ax3.set_xlabel('Amplitude [deg]')



