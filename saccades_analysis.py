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

# saccades = 

peaks = (acceleration[1:-1]>acceleration[:-2]) & (acceleration[1:-1]>acceleration[2:])
peaks = np.where(np.append(False, np.append(peaks, False)))[0]
peaks = np.intersect1d(peaks, acceleration_thresholded)


# set figure
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=[None, None, 'Azimuth', 'Elevation'])

# the first subplot shows velocity and saccades 
fig.add_trace(go.Scatter(x=times, y=velocity_x, mode='lines', 
                         opacity=0.5,
                         name='Velocity Azimuth'),
                        row=1, col=1)
fig.add_trace(go.Scatter(x=times, y=velocity_y, mode='lines', 
                         opacity=.5, 
                         name='Velocity Elevation'),
                        row=1, col=1)
fig.add_trace(go.Scatter(x=times, y=velocity_magnitude, mode='lines',
                         name='Velocity magnitude'),
                        row=1, col=1)
fig.add_trace(go.Scatter(x=times[saccades], y=velocity_magnitude[saccades], mode='markers', 
                         name='Detected saccades'),
                        row=1, col=1)
fig.update_xaxes(domain=[0, 1], title_text='Time (sec)', row=1, col=1)
fig.update_yaxes(title_text='Velocity (deg/sec)', row=1, col=1)
fig.update_layout(title=go.layout.Title(text='Detected saccades', x=0.5, y=0.9))


# add azimuth and elevation
fig.add_trace(go.Scatter(x=times, y=gaze['azimuth [deg]'], mode='lines'),
              row=2, col=1)
fig.add_trace(go.Scatter(x=times, y=gaze['elevation [deg]'], mode='lines'), 
              row=2, col=2)
fig.update_xaxes(title_text='Time (sec)', row=3, col=1)
fig.update_yaxes(title_text='Visual Angle (deg)', row=3, col=1)
fig.update_xaxes(title_text='Time (sec)', row=3, col=2)

# add velocity
fig.add_trace(go.Scatter(x=times, y=np.gradient(gaze['azimuth [deg]']), mode='lines'), 
              row=2, col=1)
fig.add_trace(go.Scatter(x=times, y=np.gradient(gaze['elevation [deg]']), mode='lines'), 
              row=2, col=1)
fig.update_yaxes(title_text='Velocity (deg/sec)', row=2, col=1)


t_gaze = go.Scatter(x=times, y=gaze['azimuth [deg]'], mode='lines')
fig.add_trace(t_gaze, row=2, col=1)
for x_min, x_max in zip(fixation_start, fixation_end):
    fig.add_shape(type='rect', x0=times[x_min], y0=0, x1=times[x_max], y1=1, 
                      fillcolor='rgba(125, 125, 125, 0.3)', 
                      line=dict(width=0), 
                      layer='below', row=2, col=1)
fig.update_xaxes(title_text='Time (sec)', row=2, col=1)
fig.update_yaxes(title_text='azimuth (deg)', row=2, col=1)

# set figy and add the trace in the figocity = np.gradient(gaze['azimuth [deg]'])
t_velocity = go.Scatter(x=times, y=velocity, mode='lines')
fig.add_trace(t_velocity, row=1, col=1)
for x_min, x_max in zip(fixation_start, fixation_end):
    fig.add_shape(type='rect', x0=times[x_min], y0=0, x1=times[x_max], y1=.5, 
                      fillcolor='rgba(125, 125, 125, 0.3)', 
                      line=dict(width=0), 
                      layer='below', row=1, col=1)
fig.update_yaxes(title_text='velocity (deg/sec)', row=1, col=1)
fig.update_layout(showlegend=False)
fig.show()

