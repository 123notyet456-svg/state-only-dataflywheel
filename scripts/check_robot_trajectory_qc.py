from lerobot.datasets import LeRobotDatasetMetadata
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

REPO_ID = "lerobot/aloha_static_cups_open"
EPISODE_INDEX = 0

print("=" * 75)
print("Robot Trajectory QC")
print("=" * 75)

# =========================
# Load metadata
# =========================

meta = LeRobotDatasetMetadata(REPO_ID)

expected_fps = meta.fps
expected_dt = 1 / expected_fps

motor_names = meta.features[
    "observation.state"
]["names"]["motors"]

expected_dim = len(motor_names)

print(f"Dataset       : {REPO_ID}")
print(f"Episode       : {EPISODE_INDEX}")
print(f"FPS           : {expected_fps}")
print(f"Expected dt   : {expected_dt:.4f} s")
print(f"State/Action  : {expected_dim} dimensions")

# =========================
# Load parquet
# =========================

relative_path = meta.get_data_file_path(EPISODE_INDEX)

local_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=str(relative_path).replace("\\", "/"),
    revision=meta.revision,
)

df = pd.read_parquet(local_path)

episode_df = df[
    df["episode_index"] == EPISODE_INDEX
].reset_index(drop=True)

print()
print(f"Frames        : {len(episode_df)}")

issues = []

# =========================
# 1. Frame continuity
# =========================

frame_indices = episode_df["frame_index"].to_numpy()

expected_frames = np.arange(len(episode_df))

if np.array_equal(frame_indices, expected_frames):
    print("Frame Index   : PASS")
else:
    print("Frame Index   : FAIL")
    issues.append("frame_index_not_continuous")

# =========================
# 2. Timestamp continuity
# =========================

timestamps = episode_df["timestamp"].to_numpy()

dt = np.diff(timestamps)

mean_dt = np.mean(dt)
max_dt = np.max(dt)
min_dt = np.min(dt)

print()
print("Timestamp Check")
print("-" * 40)

print(f"Mean dt       : {mean_dt:.6f} s")
print(f"Min dt        : {min_dt:.6f} s")
print(f"Max dt        : {max_dt:.6f} s")

timestamp_tolerance = 0.005

if np.all(
    np.abs(dt - expected_dt) < timestamp_tolerance
):
    print("Timestamp     : PASS")
else:
    print("Timestamp     : REVIEW")
    issues.append("timestamp_irregular")

# =========================
# 3. State / Action QC
# =========================

state_dim_errors = 0
action_dim_errors = 0

state_nan_count = 0
action_nan_count = 0

for _, row in episode_df.iterrows():

    state = np.asarray(row["observation.state"])
    action = np.asarray(row["action"])

    if len(state) != expected_dim:
        state_dim_errors += 1

    if len(action) != expected_dim:
        action_dim_errors += 1

    state_nan_count += np.isnan(state).sum()
    action_nan_count += np.isnan(action).sum()

print()
print("State / Action Check")
print("-" * 40)

print(f"State dim errors  : {state_dim_errors}")
print(f"Action dim errors : {action_dim_errors}")

print(f"State NaN count   : {state_nan_count}")
print(f"Action NaN count  : {action_nan_count}")

if (
    state_dim_errors == 0
    and action_dim_errors == 0
    and state_nan_count == 0
    and action_nan_count == 0
):
    print("State/Action      : PASS")
else:
    print("State/Action      : FAIL")
    issues.append("state_action_issue")

# =========================
# 4. Episode duration
# =========================

duration = timestamps[-1] - timestamps[0]

print()
print(f"Episode Duration : {duration:.2f} s")

# =========================
# 5. Done flag
# =========================

if "next.done" in episode_df.columns:

    done_values = episode_df["next.done"].to_numpy()

    done_count = np.sum(done_values)

    print(f"Done=True count  : {done_count}")

    if done_values[-1]:
        print("Episode End      : PASS")
    else:
        print("Episode End      : REVIEW")
        issues.append("final_done_not_true")

# =========================
# Final Summary
# =========================

print()
print("=" * 75)
print("QC Summary")
print("=" * 75)

if len(issues) == 0:

    print("Result : PASS ✅")

else:

    print("Result : REVIEW / FAIL")

    for issue in issues:
        print(f"- {issue}")

print()
print("Robot Trajectory QC complete ✅")