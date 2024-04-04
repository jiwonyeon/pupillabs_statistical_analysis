import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlation_lags
from scipy.signal import correlate, savgol_filter
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def get_cam2_to_cam1(imu1_to_world, imu2_to_world, angle=-102, out_format='euler'):
    
    imu_to_camera = R.from_euler('x', angle, degrees=True)

    # cam2_to_cam1 = imu_to_camera.inv()*imu2_to_world*imu1_to_world.inv()*imu_to_camera
    cam2_to_cam1 = imu_to_camera*imu1_to_world.inv()*imu2_to_world*imu_to_camera.inv()
    
    # Convert to Euler angles if requested
    if out_format == 'euler':
        # Convert to Euler angles (in 'xyz' sequence)
        euler_angles = cam2_to_cam1.as_euler('xyz')
        return euler_angles
    elif out_format == 'quat':
        # Convert to quaternion
        quat = cam2_to_cam1.as_quat()
        return quat
    elif out_format == 'matrix':
        # Convert to rotation matrix
        rot_matrix = cam2_to_cam1.as_matrix()
        return rot_matrix

def get_imu_eye(path, lowpass_filter=True):

    gaze_path = os.path.join(path, 'gaze.csv')
    imu_path = os.path.join(path, 'imu.csv')

    if os.path.exists(gaze_path) and os.path.exists(imu_path):

        print("Loading data from: " + path)

        # Read gaze, imu, and video timestamps data
        gaze_data = pd.read_csv(gaze_path)
        imu_data = pd.read_csv(imu_path)

        fixation_timestamps = gaze_data['timestamp [ns]'].values
        imu_timestamps = imu_data['timestamp [ns]'].values
        imu_to_world_quat = imu_data[['quaternion x', 'quaternion y', 'quaternion z', 'quaternion w']].values
        cam_to_eye_euler = gaze_data[['elevation [deg]', 'azimuth [deg]']].values

        # get the average sampling rate
        sr = 1e9 / np.mean(np.diff(fixation_timestamps))

        if lowpass_filter:
            # low pass filter the gaze data
            # Savitzky-Golay filter parameters
            window_length = 55 # Must be odd
            polynomial_order = 3

            # Apply the filter
            x = cam_to_eye_euler[:, 0]
            y = cam_to_eye_euler[:, 1]
            x_filtered = savgol_filter(x, window_length, polynomial_order)
            y_filtered = savgol_filter(y, window_length, polynomial_order)
            cam_to_eye_euler = np.concatenate([x_filtered.reshape(-1, 1), y_filtered.reshape(-1, 1)], axis=1)
        
        # check that timestamps are in increasing order
        if not np.all(np.diff(imu_timestamps) > 0) or not np.all(np.diff(fixation_timestamps) > 0):
            print("imu timestamps not in increasing order")
            raise ValueError("imu timestamps not in increasing order")

        # check the highest timestamp value for each
        if  fixation_timestamps[-1] > imu_timestamps[-1]:
            # truncate the longer one making sure fixation_timestamps is always <= imu_timestamps
            trunc_condition = fixation_timestamps <= imu_timestamps[-1]
            fixation_timestamps = fixation_timestamps[trunc_condition]
            cam_to_eye_euler = cam_to_eye_euler[trunc_condition]
            assert fixation_timestamps[-1] <= imu_timestamps[-1]

        # do the same for the smallest timestamp value making sure fixation_timestamps is always >= imu_timestamps
        if fixation_timestamps[0] < imu_timestamps[0]:
            trunc_condition = fixation_timestamps >= imu_timestamps[0]
            fixation_timestamps = fixation_timestamps[trunc_condition]
            cam_to_eye_euler = cam_to_eye_euler[trunc_condition]
            assert fixation_timestamps[0] >= imu_timestamps[0]
                
        # resample imu to match gaze timestamps
        imu_to_world = R.from_quat(imu_to_world_quat)
        slerp = Slerp(imu_timestamps, imu_to_world)
        imu_to_world = slerp(fixation_timestamps)

        # get deltatime between adjacent rotations
        ts2 = fixation_timestamps[1:]
        ts1 = fixation_timestamps[:-1]
        deltatime = (ts2 - ts1) / 1e9

        # get delta rotation between adjacent rotations
        imu1_to_world = imu_to_world[:-1]
        imu2_to_world = imu_to_world[1:]
        cam2_to_cam1_euler = get_cam2_to_cam1(imu1_to_world, imu2_to_world, out_format='euler')
        cam2_to_cam1_euler = cam2_to_cam1_euler / deltatime[:, None]

        # get delta eye rotation between adjacent rotations
        cam_to_eye1_euler = cam_to_eye_euler[:-1]
        cam_to_eye2_euler = cam_to_eye_euler[1:]

        # append zeros to the end of both cam_to_eye_euler (adds a z-axis rotation of 0 degrees)
        cam_to_eye1_euler_3d = np.concatenate([cam_to_eye1_euler, np.zeros((len(cam_to_eye1_euler), 1))], axis=1)
        cam_to_eye2_euler_3d = np.concatenate([cam_to_eye2_euler, np.zeros((len(cam_to_eye2_euler), 1))], axis=1)

        # get delta eye rotation between adjacent rotations using scipy rotation objects
        # create rotation objects
        cam_to_eye1 = R.from_euler('xyz', cam_to_eye1_euler_3d, degrees=True)
        cam_to_eye2 = R.from_euler('xyz', cam_to_eye2_euler_3d, degrees=True)
        # get delta eye rotation
        cam2_to_cam1_eye = cam_to_eye2*cam_to_eye1.inv()
        cam2_to_cam1_eye_euler = cam2_to_cam1_eye.as_euler('xyz', degrees=True)
        cam2_to_cam1_eye_euler = cam2_to_cam1_eye_euler / deltatime[:, None]

        print("imu and gaze data processed successfully")

    return cam2_to_cam1_euler, cam2_to_cam1_eye_euler, fixation_timestamps, sr

def plot_imu_gaze_correlation(imu, gaze, sr, save_dir=None):

    fig, axs = plt.subplots(1, 2, figsize=(2 * 13, 6))

    # Setting the overall title
    fig.suptitle('cross-correlation between IMU (deg/s) and not-lowpass filtered eye (deg/s)', fontsize=30)

    # for i in range(num_sessions):
    for j, axis in enumerate(['Elevation', 'Azimuth']):
        # Generate signals for Elevation or Azimuth
        signal1 = imu[:, j]
        signal2 = gaze[:, j]

        # Perform cross-correlation
        cross_corr = correlate(signal1, signal2, mode='full')

        # Define the time lag
        lags = correlation_lags(len(signal1), len(signal2), mode='full') * 1000 / sr  # in ms

        # Plotting the signals
        ax = axs[j]
        ax.plot(lags, cross_corr)
        ax.set_ylabel('cross-correlation', fontsize=20)
        ax.set_xlabel('Lag (ms)', fontsize=20)
        ax.set_xlim([-1500, 1500])
        # ax.set_ylim([-2e6, 2e6])
        ax.axhline(0, color='r', linestyle='--', alpha=0.3, lw=0.8)
        ax.axvline(0, color='r', linestyle='--', alpha=0.3, lw=0.8)

        # change the y-axis ticks to be exponential
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Adding Azimuth and Elevation titles only to the top row
        ax.set_title(f"{axis}\neye preceeds <------> imu preceeds", fontsize=30, pad=10)
        # Adding recording session labels to the leftmost subplots
        if j == 0:
            ax.text(-0.25, 0.5, f'Recording Session {1}', va='center', ha='center', rotation='vertical', transform=ax.transAxes, fontsize=25)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir + 'imu_gaze_correlation.png')
        print(f'imu gaze correlation plot saved to: {os.path.abspath(save_dir + "imu_gaze_correlation.png")}')
    plt.show()

def main():
    # change this to the root directory of your dataset (where you can find .mp4 and .csv files)
    root_dir = '/mnt/fs2/durango/pupil_labs_data_2023-08-01/2023-06-29_11-12-53-4a143672/'
    save_dir = './'
    
    # get the imu and gaze data
    imu, gaze, fixation_timestamps, sr = get_imu_eye(root_dir)

    # plot the correlation between the imu and gaze data
    plot_imu_gaze_correlation(imu, gaze, sr, save_dir=save_dir)

if __name__ == '__main__':
    main()