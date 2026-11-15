"""
check_label_differences.py

Diagnostic: prints how much the nearest-neighbor angle labels change from one frame to the next in
one ET - Head Mounted recording. Not part of the pipeline.

Dataset     : ET - Head Mounted (recording 004_LS47/trial_1_fcn)
Run on      : Windows workstation at TU Wien

What it does
- Reads paired_frames.csv of one recording (output of step 09; the path is hard-coded).
- Computes the frame-to-frame differences of roll_nn, pitch_nn and yaw_nn.
- Prints the standard deviation of each difference series, to give an idea of the typical size of
  the orientation change between consecutive frames (about 33 ms apart).

Notes
- The differences are plain differences; a yaw jump across +/-180 degrees is not wrapped.
"""
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