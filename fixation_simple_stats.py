import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob, json
import subprocess, pickle

"""
    This code generates some basic statistical figures for fixations
    This code uses some helper functions in pupillabs_util repository (https://github.com/grustanford/pupillabs_util)

    jiwon yeon, 2024
"""

# set directories
dataList = glob.glob('./data/2023*')
pupil_to_pkl = '../pupillabs_util/pupil_to_pkl.py'
if not os.path.exists(pupil_to_pkl):
    raise ValueError("pupil_to_pkl directory does not exist")
fig_path = './figure'

# set tasks and associated colors 
task_names = ['lego', 'walking', 'visual search', 'reading', 'watching']
colors = {
    'lego': sns.light_palette("seagreen", 5),
    'walking': sns.light_palette("blue", 5), 
    'visual search': sns.light_palette("red", 5),
    'reading': sns.light_palette("orange", 5),
    'watching': sns.light_palette("purple", 5)
    }

# pre-set custom legend handles 
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['lego'][3], markersize=10, label='Lego'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['reading'][3], markersize=10, label='reading'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['visual search'][3], markersize=10, label='visual search'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['walking'][3], markersize=10, label='walking'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['watching'][3], markersize=10, label='watching'),
]

# open figures
fig1, ax1 = plt.subplots() # first figure shows histogram of fixation duration
fig2, ax2 = plt.subplots() # second figure shows individual heatmap
fig3, ax3 = plt.subplots() # third figure shows distance between fixations 
fig4, ax4 = plt.subplots() # fourth figure shows the number of fixations within a second
fig5, ax5 = plt.subplots() # fifth figure shows gaze angle relative to gravity 
    
for idx in range(len(dataList)): 
    dataPath = os.path.abspath(dataList[idx])       # feed in the absolute path of the data direcotry

    # generate pkl file 
    command = ["python", pupil_to_pkl, dataPath]
    if not os.path.exists(os.path.join(dataPath, 'eyedata.pkl')):
        subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    pupil_data = pickle.load(open(os.path.join(dataPath, 'eyedata.pkl'), 'rb'))

    # match the recording's task to the task names
    if "waldo" in str(np.squeeze(pupil_data['note'])).lower():
        pupil_data['note'] = 'visual search'
        thiscolor = colors['visual search']
    elif "walking" in str(np.squeeze(pupil_data['note'])).lower():
        pupil_data['note'] = 'walking'
        thiscolor = colors['walking']
    elif "lego" in str(np.squeeze(pupil_data['note'])).lower():
        pupil_data['note'] = 'lego'
        thiscolor = colors['lego']
    elif "reading" in str(np.squeeze(pupil_data['note'])).lower():
        pupil_data['note'] = 'reading'
        thiscolor = colors['reading']
    elif "watching" in str(np.squeeze(pupil_data['note'])).lower():
        pupil_data['note'] = 'watching'
        thiscolor = colors['watching']


    # select fixation values that only needs to be considered - while actively doing the task 
    if not pupil_data['events'].empty:
        events = pupil_data['fixation']['event name'].dropna()
        event_start = events[events.str.contains('start|begin')].index
        event_end = events[events.str.contains('end|finish')].index
        fixation = pd.DataFrame()
        for start_index, end_index in zip(event_start, event_end):
            event_rows = pupil_data['fixation'].loc[start_index:end_index]
            fixation = pd.concat([fixation, event_rows])
    else:
        fixation = pupil_data['fixation']

    # Fig1. Histogram of fixation duration
    bin_edges = np.linspace(0, 2000, 100)
    ax1.hist(fixation['duration [ms]'], bins = bin_edges, alpha = .3, color = thiscolor[3])
    if idx == len(dataList)-1:
        ax1.set_xlabel('Duration [ms]')
        ax1.set_ylabel('Count')
        ax1.legend(handles = legend_handles)
        ax1.set_title('Fixation duration')
        fig1.savefig(os.path.join(fig_path, 'fixation_duration.png'))

    # Fig2. Individual heatmap 
    ax2 = fig2.subplots()
    ax2.hist2d(fixation['azimuth [deg]'], fixation['elevation [deg]'], bins=(100,100), cmap = 'viridis')
    ax2.invert_yaxis()
    ax2.set_xlabel('Visual angle [deg]')
    ax2.set_ylabel('visual angle [deg]')
    ax2.set_title(f'Fixation heat map: {pupil_data["note"]}-{pupil_data["wearer"]}')
    fig_name = f'heatmap_{pupil_data["note"]}_{pupil_data["wearer"]}.png'
    fig2.savefig(os.path.join(fig_path, fig_name))
    fig2.clf()

    # Fig3. Distance between sequential fixations
    distance = np.linalg.norm(fixation[['azimuth [deg]', 'elevation [deg]']].iloc[:-1].to_numpy()-fixation[['azimuth [deg]', 'elevation [deg]']].iloc[1:].to_numpy(), axis=1)
    ax3.hist(distance, bins=np.linspace(0,50,100), alpha=.3, color = thiscolor[3])
    if idx == len(dataList)-1:
        ax3.set_xlabel('Distance [deg]')
        ax3.set_ylabel('Count')
        ax3.set_title('Distance between fixations')
        ax3.legend(handles = legend_handles)
        fig3.savefig(os.path.join(fig_path, 'fixation_distance.png'))

    # Fig4. Number of fixations with in a second
    n_fixation = []
    duration = 0
    count = 0 
    for id in range(len(fixation)):
        duration += fixation['duration [ms]'].iloc[id]
        count += 1
        if duration > 1000:
            n_fixation.append(count)
            count = 0 
            duration = 0 
        elif id == len(fixation)-1:
            n_fixation.append(count)

    ax4.hist(n_fixation, bins = np.arange(1,11), alpha = .3, color = thiscolor[3])
    if idx == len(dataList)-1:
        ax4.set_xlabel('Number of fixations')
        ax4.set_ylabel('Count')
        ax4.set_title('Fixation count per second')
        ax4.legend(handles = legend_handles)
        fig4.savefig(os.path.join(fig_path, 'fixation_count_per_second.png'))

    # Fig5. Gaze angle relative to gravity
    # compute gave angle relative to gravity
    getgazePath = '../pupillabs_util/gaze_angle_relative_to_gravity.py'
    command = ["python", getgazePath, dataPath]
    if not os.path.exists(os.path.join(dataPath, 'gaze_angle_relative_to_gravity.csv')):
        subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    gaze_angles = pd.read_csv(os.path.join(dataPath, 'gaze_angle_relative_to_gravity.csv'))

    # display histogram
    ax5.hist(gaze_angles, bins=50, alpha=0.3, color=thiscolor[3])
    if idx == len(dataList)-1:
        ax5.set_xlabel('Angle [deg]')
        ax5.set_ylabel('Count')
        ax5.set_title('Gaze angle relative to gravity')
        ax5.legend(handles = legend_handles)
        fig5.savefig(os.path.join(fig_path, 'gaze_angle_relative_to_gravity.png'))


