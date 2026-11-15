"""
play_video_with_angles.py

Diagnostic: plays a scene-camera video next to the synchronized IMU angles, to check the
synchronization visually. Not part of the pipeline.

Dataset     : ET - Upper Body (explored, not used for the final model)
Run on      : Windows workstation at TU Wien
Recording   : participant p16, leg 2 (paths are hard-coded in the CONFIG section)

What it does
- Loads `paired_frames.csv` of one participant/leg (output of step 08) and the matching world.mp4.
- Converts the frame timestamps to seconds, applies a manual shift (IMU_TO_VIDEO_OFFSET_S) and
  starts the time axis at 0 so that it matches the video time.
- Opens an interactive matplotlib window with:
    the video frame, three time-series plots (roll, pitch, yaw of the external IMU) with a cursor
    at the current time, and a bar chart with the current values.
- Plays the video at PLAYBACK_SPEED and moves the cursors with it. If head movements in the video
  and changes in the curves do not happen at the same time, the synchronization is off.

Notes
- IMU_TO_VIDEO_OFFSET_S was tuned by hand for this recording.
- The variable IMU_TIME_COL_MS holds the column `frame_ts_ns`, which is in nanoseconds
  (it is converted with * 1e-9).
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG (EDIT THESE)
# -----------------------------
VIDEO_PATH = r"E:\john\ET-IMU-and-upper-body\et\p16\2024-08-08_19-08-02-4b603a09\world.mp4"
IMU_CSV_PATH = r"D:\upper_body_sync_2\paired_frames\p16\leg_2\paired_frames.csv"

# IMU time column (YOU HAVE ms -> milliseconds)
IMU_TIME_COL_MS = "frame_ts_ns"

ROLL_COL  = "roll_external_nn"
PITCH_COL = "pitch_external_nn"
YAW_COL   = "yaw_external_nn"

# If IMU appears delayed/advanced relative to video, tweak this (seconds)
# Positive -> shifts IMU forward in time axis relative to video
IMU_TO_VIDEO_OFFSET_S = -145.86922991373905

# Plot view: show full duration OR only first N seconds
SHOW_FULL_DURATION = True
SHOW_FIRST_SECONDS = 60.0  # used only if SHOW_FULL_DURATION=False

# Playback speed (1.0 = real-time-ish; increase if laggy)
PLAYBACK_SPEED = 20.0

# -----------------------------
# FILE CHECKS
# -----------------------------
print("Video exists:", os.path.exists(VIDEO_PATH))
print("IMU exists:", os.path.exists(IMU_CSV_PATH))
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(VIDEO_PATH)
if not os.path.exists(IMU_CSV_PATH):
    raise FileNotFoundError(IMU_CSV_PATH)

# -----------------------------
# LOAD IMU
# -----------------------------
df = pd.read_csv(IMU_CSV_PATH)

t_s = df[IMU_TIME_COL_MS].astype(np.float64).to_numpy() * 1e-9 #ns -> seconds
t_s = t_s + IMU_TO_VIDEO_OFFSET_S
t_s = t_s - t_s[0]  # IMPORTANT: start IMU time at 0 so it matches video time axis

roll  = df[ROLL_COL].astype(np.float64).to_numpy()
pitch = df[PITCH_COL].astype(np.float64).to_numpy()
yaw   = df[YAW_COL].astype(np.float64).to_numpy()

print("IMU t_s range:", float(t_s.min()), float(t_s.max()))

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 20.0
dt = 1.0 / fps

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration_s = total_frames / fps
print(f"FPS: {fps:.3f}, frames: {total_frames}, duration: {video_duration_s:.3f}s")

# Read first frame for display init
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame.")
frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

# -----------------------------
# HELPER: find nearest IMU index for a given video time
# -----------------------------
def nearest_idx(time_array, t):
    # time_array must be sorted (it should be)
    i = int(np.searchsorted(time_array, t))
    if i <= 0:
        return 0
    if i >= len(time_array):
        return len(time_array) - 1
    # choose closer between i-1 and i
    return i if abs(time_array[i] - t) < abs(time_array[i-1] - t) else i - 1

# -----------------------------
# SETUP FIGURE (VIDEO + 3 PLOTS + 4th "live values")
# -----------------------------
plt.ion()
fig = plt.figure(figsize=(4, 9))
gs = fig.add_gridspec(5, 1, height_ratios=[3.0, 1.0, 1.0, 1.0, 1.2], hspace=0.35)

ax_vid = fig.add_subplot(gs[0])
ax_r   = fig.add_subplot(gs[1])
ax_p   = fig.add_subplot(gs[2], sharex=ax_r)
ax_y   = fig.add_subplot(gs[3], sharex=ax_r)
ax_bar = fig.add_subplot(gs[4])

im = ax_vid.imshow(frame0_rgb)
ax_vid.set_title("Space: pause/play | Left/Right: step frame | Esc: quit")
ax_vid.axis("off")

# Time-series lines
ax_r.plot(t_s, roll, linewidth=1)
ax_p.plot(t_s, pitch, linewidth=1)
ax_y.plot(t_s, yaw, linewidth=1)

ax_r.set_ylabel("roll (deg)")
ax_p.set_ylabel("pitch (deg)")
ax_y.set_ylabel("yaw (deg)")
ax_y.set_xlabel("time (s)")

# Fix x-limits (so the line MOVES)
if SHOW_FULL_DURATION:
    ax_y.set_xlim(0, video_duration_s)
else:
    ax_y.set_xlim(0, min(SHOW_FIRST_SECONDS, video_duration_s))

# Cursors: vertical time line + moving dot + horizontal current-value line
cursor_r_t = ax_r.axvline(x=0.0, linewidth=1)
cursor_p_t = ax_p.axvline(x=0.0, linewidth=1)
cursor_y_t = ax_y.axvline(x=0.0, linewidth=1)

dot_r, = ax_r.plot([0.0], [roll[0]], marker='o', markersize=6, linestyle='None')
dot_p, = ax_p.plot([0.0], [pitch[0]], marker='o', markersize=6, linestyle='None')
dot_y, = ax_y.plot([0.0], [yaw[0]], marker='o', markersize=6, linestyle='None')

hline_r = ax_r.axhline(y=roll[0], linewidth=1, linestyle='--')
hline_p = ax_p.axhline(y=pitch[0], linewidth=1, linestyle='--')
hline_y = ax_y.axhline(y=yaw[0], linewidth=1, linestyle='--')

# Live value text (top-left on video)
val_text = ax_vid.text(
    0.02, 0.95,
    "",
    transform=ax_vid.transAxes,
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round", alpha=0.6)
)

# 4th plot: live bars
labels = ["roll", "pitch", "yaw"]
bar_vals = [roll[0], pitch[0], yaw[0]]
bars = ax_bar.bar(labels, bar_vals)
ax_bar.set_ylabel("deg (current)")
ax_bar.set_title("Current angle values (live)")

# reasonable y-limits for bars based on data range (robust)
def robust_ylim(x):
    lo = np.nanpercentile(x, 1)
    hi = np.nanpercentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    pad = 0.15 * (hi - lo + 1e-6)
    return lo - pad, hi + pad

r_lo, r_hi = robust_ylim(roll)
p_lo, p_hi = robust_ylim(pitch)
y_lo, y_hi = robust_ylim(yaw)

# Keep each time-series auto-scaled nicely
ax_r.set_ylim(r_lo, r_hi)
ax_p.set_ylim(p_lo, p_hi)
ax_y.set_ylim(y_lo, y_hi)

# Bar chart y-limits that can hold all 3 reasonably
bar_lo = min(r_lo, p_lo, y_lo)
bar_hi = max(r_hi, p_hi, y_hi)
ax_bar.set_ylim(bar_lo, bar_hi)

# -----------------------------
# PLAYBACK CONTROLS
# -----------------------------
paused = False
frame_idx = 0

def draw_frame(idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        return False
    fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    im.set_data(fr_rgb)
    return True

def update_visuals(video_t):
    i = nearest_idx(t_s, video_t)

    r = float(roll[i])
    p = float(pitch[i])
    yv = float(yaw[i])

    # vertical time cursor
    cursor_r_t.set_xdata([video_t, video_t])
    cursor_p_t.set_xdata([video_t, video_t])
    cursor_y_t.set_xdata([video_t, video_t])

    # moving dot (t, value)
    dot_r.set_data([video_t], [r])
    dot_p.set_data([video_t], [p])
    dot_y.set_data([video_t], [yv])

    # horizontal line at current value
    hline_r.set_ydata([r, r])
    hline_p.set_ydata([p, p])
    hline_y.set_ydata([yv, yv])

    # update text
    val_text.set_text(f"t = {video_t:6.2f}s\nroll  = {r:8.2f}°\npitch = {p:8.2f}°\nyaw   = {yv:8.2f}°")

    # update bars
    bars[0].set_height(r)
    bars[1].set_height(p)
    bars[2].set_height(yv)

def on_key(event):
    global paused, frame_idx
    if event.key == ' ':
        paused = not paused
    elif event.key == 'escape':
        plt.close(fig)
    elif event.key == 'right':
        paused = True
        frame_idx = min(total_frames - 1, frame_idx + 1)
        draw_frame(frame_idx)
        update_visuals(frame_idx * dt)
    elif event.key == 'left':
        paused = True
        frame_idx = max(0, frame_idx - 1)
        draw_frame(frame_idx)
        update_visuals(frame_idx * dt)

fig.canvas.mpl_connect('key_press_event', on_key)

# init visuals at t=0
update_visuals(0.0)
fig.canvas.draw()
fig.canvas.flush_events()

# -----------------------------
# MAIN LOOP
# -----------------------------
while plt.fignum_exists(fig.number):
    if not paused:
        frame_idx += 1
        if frame_idx >= total_frames:
            break
        if not draw_frame(frame_idx):
            break

    video_t = frame_idx * dt
    update_visuals(video_t)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(max(0.001, dt / PLAYBACK_SPEED))

cap.release()
plt.ioff()
plt.show()