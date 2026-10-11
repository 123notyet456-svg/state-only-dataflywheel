from lerobot.datasets import LeRobotDataset


REPO_ID = "lerobot/aloha_static_cups_open"

print("=" * 60)
print("Dataset v1 - Read Check")
print("=" * 60)

dataset = LeRobotDataset(REPO_ID)

print("Repo ID:", REPO_ID)
print("Total Episodes:", dataset.meta.total_episodes)
print("Total Frames:", len(dataset))

sample = dataset[0]

print("\nFirst Frame:")
print("episode_index:", sample["episode_index"])
print("frame_index:", sample["frame_index"])
print("timestamp:", sample["timestamp"])

print("\nState shape:", sample["observation.state"].shape)
print("Action shape:", sample["action"].shape)

print("\nState:")
print(sample["observation.state"])

print("\nAction:")
print(sample["action"])

import csv
from pathlib import Path

report_path = Path("reports/episode_frame_summary.csv")
report_path.parent.mkdir(parents=True, exist_ok=True)

rows = []

for ep_idx in range(dataset.meta.total_episodes):
    start_idx = int(dataset.meta.episodes["dataset_from_index"][ep_idx])
    end_idx = int(dataset.meta.episodes["dataset_to_index"][ep_idx])
    num_frames = end_idx - start_idx

    rows.append([
        ep_idx,
        start_idx,
        end_idx,
        num_frames,
    ])

with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow([
        "episode_index",
        "dataset_from_index",
        "dataset_to_index",
        "num_frames",
    ])

    writer.writerows(rows)

print("\nEpisode Frame Summary:")
for row in rows:
    print(
        f"Episode {row[0]:02d}: "
        f"{row[3]} frames "
        f"[{row[1]} -> {row[2]})"
    )

print("\nReport saved to:", report_path)

split_report_path = Path("reports/dataset_split.csv")

split_rows = []

for ep_idx in range(dataset.meta.total_episodes):
    if ep_idx < 40:
        split = "train"
    elif ep_idx < 45:
        split = "val"
    else:
        split = "test"

    start_idx = int(dataset.meta.episodes["dataset_from_index"][ep_idx])
    end_idx = int(dataset.meta.episodes["dataset_to_index"][ep_idx])
    num_frames = end_idx - start_idx

    split_rows.append([
        ep_idx,
        split,
        start_idx,
        end_idx,
        num_frames,
    ])

with open(split_report_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow([
        "episode_index",
        "split",
        "dataset_from_index",
        "dataset_to_index",
        "num_frames",
    ])

    writer.writerows(split_rows)

print("\nDataset Split:")
print("Train: Episode 00-39")
print("Val:   Episode 40-44")
print("Test:  Episode 45-49")
print("\nSplit report saved to:", split_report_path)

train_eps = set(range(0, 40))
val_eps = set(range(40, 45))
test_eps = set(range(45, 50))

train_val_overlap = train_eps & val_eps
train_test_overlap = train_eps & test_eps
val_test_overlap = val_eps & test_eps

total_eps = len(train_eps | val_eps | test_eps)
total_frames = sum(row[4] for row in split_rows)

print("\nSplit QC:")
print("Train-Val overlap:", train_val_overlap)
print("Train-Test overlap:", train_test_overlap)
print("Val-Test overlap:", val_test_overlap)
print("Total Episodes:", total_eps)
print("Total Frames:", total_frames)

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
    and total_eps == 50
    and total_frames == 20000
):
    print("Split QC: PASS")
else:
    print("Split QC: FAIL")

selected_dataset = dataset.select_columns([
    "episode_index",
    "frame_index",
    "timestamp",
    "observation.state",
    "action",
])

print("\nSelected Columns:")
print(selected_dataset.column_names)

selected_sample = selected_dataset[0]

print("\nSelected Sample:")
print("episode_index:", selected_sample["episode_index"])
print("frame_index:", selected_sample["frame_index"])
print("timestamp:", selected_sample["timestamp"])
print("State shape:", selected_sample["observation.state"].shape)
print("Action shape:", selected_sample["action"].shape)

import numpy as np

# ============================================================
# Build Train Split
# ============================================================

train_dataset = selected_dataset.select(range(0, 16000))
train_dataset = train_dataset.with_format("numpy")

train_data = train_dataset[:]

train_state = np.asarray(train_data["observation.state"])
train_action = np.asarray(train_data["action"])
train_episode = np.asarray(train_data["episode_index"])
train_frame = np.asarray(train_data["frame_index"])
train_timestamp = np.asarray(train_data["timestamp"])

print("\nTrain Array Shapes:")
print("state:", train_state.shape)
print("action:", train_action.shape)
print("episode_index:", train_episode.shape)
print("frame_index:", train_frame.shape)
print("timestamp:", train_timestamp.shape)

np.savez(
    "dataset_v1/train.npz",
    state=train_state,
    action=train_action,
    episode_index=train_episode,
    frame_index=train_frame,
    timestamp=train_timestamp,
)

print("\nSaved: dataset_v1/train.npz")

# ============================================================
# Build Val and Test Splits
# ============================================================

def save_split(name, start_idx, end_idx):
    split_dataset = selected_dataset.select(range(start_idx, end_idx))
    split_dataset = split_dataset.with_format("numpy")

    split_data = split_dataset[:]

    split_state = np.asarray(split_data["observation.state"])
    split_action = np.asarray(split_data["action"])
    split_episode = np.asarray(split_data["episode_index"])
    split_frame = np.asarray(split_data["frame_index"])
    split_timestamp = np.asarray(split_data["timestamp"])

    print(f"\n{name.upper()} Array Shapes:")
    print("state:", split_state.shape)
    print("action:", split_action.shape)
    print("episode_index:", split_episode.shape)
    print("frame_index:", split_frame.shape)
    print("timestamp:", split_timestamp.shape)

    output_path = f"dataset_v1/{name}.npz"

    np.savez(
        output_path,
        state=split_state,
        action=split_action,
        episode_index=split_episode,
        frame_index=split_frame,
        timestamp=split_timestamp,
    )

    print("Saved:", output_path)


save_split("val", 16000, 18000)
save_split("test", 18000, 20000)