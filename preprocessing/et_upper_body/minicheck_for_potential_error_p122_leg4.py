from pathlib import Path
import pandas as pd

# ---- EDIT THESE ----
LEG4 = Path(r"E:\john\ET-IMU-and-upper-body\imu\p112\leg_4_allInOne.csv")
ET_RECORDING = Path(r"E:\john\ET-IMU-and-upper-body\et\p112\2024-09-26_10-41-22-502fd253")
WORLD_TS = ET_RECORDING / "world_timestamps.csv"

# ---- LOAD ----
leg = pd.read_csv(LEG4)
wt = pd.read_csv(WORLD_TS)

# allInOne ms is epoch milliseconds -> convert to ns
leg_t0_ns = int(leg["ms"].iloc[0]) * 1_000_000
leg_t1_ns = int(leg["ms"].iloc[-1]) * 1_000_000

# world timestamps are epoch ns
wt_t0_ns = int(wt["timestamp [ns]"].iloc[0])
wt_t1_ns = int(wt["timestamp [ns]"].iloc[-1])

print("LEG4 fn unique:", leg["fn"].unique())
print("LEG4 time span (ns):", leg_t0_ns, "->", leg_t1_ns, "dur_s=", (leg_t1_ns-leg_t0_ns)/1e9)

print("WORLD time span (ns):", wt_t0_ns, "->", wt_t1_ns, "dur_s=", (wt_t1_ns-wt_t0_ns)/1e9)

# overlap check (how far apart are starts)
print("abs(start diff) seconds:", abs(leg_t0_ns - wt_t0_ns)/1e9)

# overlap ratio idea
overlap_start = max(leg_t0_ns, wt_t0_ns)
overlap_end = min(leg_t1_ns, wt_t1_ns)
overlap = max(0, overlap_end - overlap_start)
print("overlap_s:", overlap/1e9)