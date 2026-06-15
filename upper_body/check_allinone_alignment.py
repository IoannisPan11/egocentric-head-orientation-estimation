import pandas as pd
import numpy as np

df = pd.read_csv("E:\john\ET-IMU-and-head-mounted-imu\paired_frames/004_LS47/trial_1_fcn\paired_frames.csv")

roll = df["roll_nn"].values
pitch = df["pitch_nn"].values
yaw = df["yaw_nn"].values

d_roll = np.diff(roll)
d_pitch = np.diff(pitch)
d_yaw = np.diff(yaw)

print("roll std:", np.std(d_roll))
print("pitch std:", np.std(d_pitch))
print("yaw std:", np.std(d_yaw))