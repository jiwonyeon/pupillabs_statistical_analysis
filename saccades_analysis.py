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

# create figure for plotting the acceleration, velocity, and the raw data
fig = plt.figure(figsize=(15,8))
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 2), (2, 0))
ax4 = plt.subplot2grid((3, 2), (2, 1))

ax1.plot(times, acceleration, alpha=0.5, label='acceleration')
ax1.scatter(times[acceleration_thresholded], acceleration[acceleration_thresholded], 
            color='red', label='supra-thresholded')
ax1.set_ylabel(r'Acceleration[deg/sec$^{-2}$]')
ax1.legend()

ax2.plot(times, velocity_x, alpha=0.5, label='azimuth')
ax2.plot(times, velocity_y, alpha=0.5, label='elevation')
ax2.plot(times, velocity_magnitude, alpha=0.8, label='magnitude')
ax2.scatter(times[acceleration_thresholded], velocity_magnitude[acceleration_thresholded], label='supra-thresholded')
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Velocity [deg/s]')
ax2.legend(loc='lower right')

ax3.plot(times, gaze['azimuth [deg]'])
ax3.scatter(times[acceleration_thresholded], gaze['azimuth [deg]'].iloc[acceleration_thresholded], label='supra-thresholded')
ax3.set_xlabel('Time [sec]')
ax3.set_ylabel('Azimuth [deg]')

ax4.plot(times, gaze['elevation [deg]'])
ax4.scatter(times[acceleration_thresholded], gaze['elevation [deg]'].iloc[acceleration_thresholded], label='supra-thresholded')
ax4.set_xlabel('Time [sec]')
ax4.set_ylabel('Elevation [deg]')

