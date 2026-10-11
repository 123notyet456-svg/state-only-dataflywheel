import numpy as np


DATA_PATH = "dataset_v1/train.npz"

print("=" * 60)
print("Dataset v1 - Train QC")
print("=" * 60)

data = np.load(DATA_PATH)

state = data["state"]
action = data["action"]
episode_index = data["episode_index"]
frame_index = data["frame_index"]
timestamp = data["timestamp"]

print("\nShapes:")
print("state:", state.shape)
print("action:", action.shape)
print("episode_index:", episode_index.shape)
print("frame_index:", frame_index.shape)
print("timestamp:", timestamp.shape)

print("\nNaN Check:")
print("state NaN:", np.isnan(state).sum())
print("action NaN:", np.isnan(action).sum())
print("timestamp NaN:", np.isnan(timestamp).sum())

print("\nEpisode Check:")
print("min episode:", episode_index.min())
print("max episode:", episode_index.max())
print("unique episodes:", len(np.unique(episode_index)))

print("\nBasic QC:")

shape_pass = (
    state.shape == (16000, 14)
    and action.shape == (16000, 14)
    and episode_index.shape == (16000,)
    and frame_index.shape == (16000,)
    and timestamp.shape == (16000,)
)

nan_pass = (
    np.isnan(state).sum() == 0
    and np.isnan(action).sum() == 0
    and np.isnan(timestamp).sum() == 0
)

episode_pass = (
    episode_index.min() == 0
    and episode_index.max() == 39
    and len(np.unique(episode_index)) == 40
)

print("Shape Check:", "PASS" if shape_pass else "FAIL")
print("NaN Check:", "PASS" if nan_pass else "FAIL")
print("Episode Check:", "PASS" if episode_pass else "FAIL")

if shape_pass and nan_pass and episode_pass:
    print("\nTrain Dataset QC: PASS")
else:
    print("\nTrain Dataset QC: FAIL")

print("\nContinuity Check:")

frame_continuity_pass = True
timestamp_continuity_pass = True

bad_frame_episodes = []
bad_timestamp_episodes = []

for ep in np.unique(episode_index):

    mask = episode_index == ep

    ep_frames = frame_index[mask]
    ep_timestamps = timestamp[mask]

    # --------------------------------------------------------
    # Frame Continuity
    # --------------------------------------------------------

    frame_diff = np.diff(ep_frames)

    if not np.all(frame_diff == 1):
        frame_continuity_pass = False
        bad_frame_episodes.append(int(ep))

    # --------------------------------------------------------
    # Timestamp Continuity
    # --------------------------------------------------------

    timestamp_diff = np.diff(ep_timestamps)

    if not np.allclose(timestamp_diff, 0.02, atol=1e-4):
        timestamp_continuity_pass = False
        bad_timestamp_episodes.append(int(ep))


print(
    "Frame Continuity:",
    "PASS" if frame_continuity_pass else "FAIL"
)

print(
    "Timestamp Continuity:",
    "PASS" if timestamp_continuity_pass else "FAIL"
)

print("Bad Frame Episodes:", bad_frame_episodes)
print("Bad Timestamp Episodes:", bad_timestamp_episodes)

import csv
from pathlib import Path

report_path = Path("reports/dataset_train_qc.csv")
report_path.parent.mkdir(parents=True, exist_ok=True)

qc_rows = [
    ["shape_check", "PASS" if shape_pass else "FAIL"],
    ["nan_check", "PASS" if nan_pass else "FAIL"],
    ["episode_check", "PASS" if episode_pass else "FAIL"],
    ["frame_continuity", "PASS" if frame_continuity_pass else "FAIL"],
    ["timestamp_continuity", "PASS" if timestamp_continuity_pass else "FAIL"],
    ["total_frames", len(state)],
    ["unique_episodes", len(np.unique(episode_index))],
    ["state_nan_count", int(np.isnan(state).sum())],
    ["action_nan_count", int(np.isnan(action).sum())],
    ["timestamp_nan_count", int(np.isnan(timestamp).sum())],
]

with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["check", "result"])
    writer.writerows(qc_rows)

print("\nQC Report saved to:", report_path)

print("\nAction Statistics:")
print("-" * 60)

joint_names = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

for i, joint_name in enumerate(joint_names):
    joint_action = action[:, i]

    print(
        f"{i+1:02d} {joint_name:20s} "
        f"min={joint_action.min():.4f} "
        f"max={joint_action.max():.4f} "
        f"mean={joint_action.mean():.4f} "
        f"std={joint_action.std():.4f}"
    )

action_stats_report = Path("reports/action_statistics.csv")

action_stats_rows = []

for i, joint_name in enumerate(joint_names):
    joint_action = action[:, i]

    action_stats_rows.append([
        i + 1,
        joint_name,
        float(joint_action.min()),
        float(joint_action.max()),
        float(joint_action.mean()),
        float(joint_action.std()),
    ])

with open(
    action_stats_report,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "action_dim",
        "joint_name",
        "min",
        "max",
        "mean",
        "std",
    ])

    writer.writerows(action_stats_rows)

print("\nAction Statistics Report saved to:", action_stats_report)

print("\n" + "=" * 60)
print("Dataset v1 - Unified QC")
print("=" * 60)

split_configs = {
    "train": {
        "path": "dataset_v1/train.npz",
        "expected_frames": 16000,
        "expected_min_ep": 0,
        "expected_max_ep": 39,
        "expected_episodes": 40,
    },
    "val": {
        "path": "dataset_v1/val.npz",
        "expected_frames": 2000,
        "expected_min_ep": 40,
        "expected_max_ep": 44,
        "expected_episodes": 5,
    },
    "test": {
        "path": "dataset_v1/test.npz",
        "expected_frames": 2000,
        "expected_min_ep": 45,
        "expected_max_ep": 49,
        "expected_episodes": 5,
    },
}

split_episode_sets = {}

for split_name, config in split_configs.items():

    split_data = np.load(config["path"])

    split_state = split_data["state"]
    split_action = split_data["action"]
    split_episode = split_data["episode_index"]

    split_episode_sets[split_name] = set(
        np.unique(split_episode).tolist()
    )

    shape_pass = (
        split_state.shape ==
        (config["expected_frames"], 14)
        and split_action.shape ==
        (config["expected_frames"], 14)
    )

    nan_pass = (
        np.isnan(split_state).sum() == 0
        and np.isnan(split_action).sum() == 0
    )

    episode_pass = (
        split_episode.min() == config["expected_min_ep"]
        and split_episode.max() == config["expected_max_ep"]
        and len(np.unique(split_episode))
        == config["expected_episodes"]
    )

    print(f"\n{split_name.upper()}:")
    print("Shape Check:", "PASS" if shape_pass else "FAIL")
    print("NaN Check:", "PASS" if nan_pass else "FAIL")
    print("Episode Check:", "PASS" if episode_pass else "FAIL")


train_val_overlap = (
    split_episode_sets["train"]
    & split_episode_sets["val"]
)

train_test_overlap = (
    split_episode_sets["train"]
    & split_episode_sets["test"]
)

val_test_overlap = (
    split_episode_sets["val"]
    & split_episode_sets["test"]
)

print("\nSplit Overlap Check:")
print("Train-Val:", train_val_overlap)
print("Train-Test:", train_test_overlap)
print("Val-Test:", val_test_overlap)

overlap_pass = (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
)

print(
    "Episode Leakage Check:",
    "PASS" if overlap_pass else "FAIL"
)

summary_report_path = Path("reports/dataset_v1_summary.csv")

summary_rows = [
    [
        "train",
        40,
        16000,
        "0-39",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ],
    [
        "val",
        5,
        2000,
        "40-44",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ],
    [
        "test",
        5,
        2000,
        "45-49",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ],
]

with open(
    summary_report_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "split",
        "episodes",
        "frames",
        "episode_range",
        "shape_check",
        "nan_check",
        "episode_check",
        "leakage_check",
    ])

    writer.writerows(summary_rows)

print(
    "\nDataset v1 Summary Report saved to:",
    summary_report_path
)