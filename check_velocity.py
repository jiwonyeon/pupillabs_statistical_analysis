import os, glob
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

""" This code was written to check the velocity of the eye data.
    The code compared velocity between Pupil Labs' and Tobii's device. 
    While Pupil Labs' data looked noisier, when applied some filtering (savgol_filter), 
    the pupil labs' data looked also reasonable. 

    Jiwon Yeon. 2024
"""

data_path = './data'
folders = glob.glob(data_path + '/2023*')
figure_path = './figure'
device_name = 'pupil'

for folder in folders:
    # load eye data
    eyedata = pickle.load(open(folder + '/eyedata.pkl', 'rb'))

    # load recording information
    wearer = eyedata['wearer']
    task = eyedata['note'][0][0]

    # compute time difference
    times = (eyedata['gaze']['timestamp [ns]'] - eyedata['gaze']['timestamp [ns]'][0]) / 1e9
    time_diff = np.append(0, np.round(np.diff(times),5))
    fsp = round(np.average(np.diff(times)),3)

    # apply savglo filter to gaze data
    window_length = 55
    polynomial_order = 3
    azimuth = eyedata['gaze']['azimuth [deg]']
    elevation = eyedata['gaze']['elevation [deg]']
    azimuth_filtered = savgol_filter(azimuth, window_length, polynomial_order)
    elevation_filtered = savgol_filter(elevation, window_length, polynomial_order)
    amplitude = np.sqrt(azimuth_filtered**2 + elevation_filtered**2)

    # compute velocity 
    velocity = np.gradient(amplitude, fsp)

    # draw gaze point figure
    fig1, (ax1, ax2) = plt.subplots(2,1,figsize=(15,10))
    ax1.plot(times, azimuth_filtered, label='azimuth')
    ax1.plot(times, elevation_filtered, label='elevation')
    ax1.plot(times, amplitude, label='amplitude')
    ax1.legend(loc='lower right')
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Gaze angle [deg]')

    # draw velocity 
    ax2.plot(times, velocity)
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Velocity [deg/sec]')

    fig1.suptitle(f'{device_name} - {wearer} - {task}')
    fig1.savefig(os.path.join(figure_path, f'{device_name}_{wearer}_{task}.png'))
    
    
