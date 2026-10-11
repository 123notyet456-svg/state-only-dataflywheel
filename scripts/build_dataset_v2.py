from pathlib import Path
import csv
import shutil
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "dataset_v1"
OUTPUT_DIR = PROJECT_ROOT / "dataset_v2"
REPORT_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = INPUT_DIR / "train.npz"
VAL_PATH = INPUT_DIR / "val.npz"
TEST_PATH = INPUT_DIR / "test.npz"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


TOP_PERCENT = 10
EXTRA_COPIES = 2


def find_key(data, candidates):
    for key in candidates:
        if key in data.files:
            return key

    raise KeyError(
        f"Cannot find required key.\n"
        f"Candidates: {candidates}\n"
        f"Available keys: {data.files}"
    )


print("=" * 70)
print("Dataset v2 Construction")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load Dataset v1 Train
# ------------------------------------------------------------

data = np.load(TRAIN_PATH, allow_pickle=True)

print("\nDataset v1 keys:")
for key in data.files:
    print(f"  {key}: {data[key].shape}")


action_key = find_key(
    data,
    [
        "action",
        "actions",
    ],
)

episode_key = find_key(
    data,
    [
        "episode_index",
        "episode_indices",
        "episode",
    ],
)

# frame_index is useful for reporting, but not required for construction
frame_key = None

for candidate in [
    "frame_index",
    "frame_indices",
    "frame",
]:
    if candidate in data.files:
        frame_key = candidate
        break


actions = np.asarray(data[action_key])
episode_indices = np.asarray(data[episode_key]).reshape(-1)

n_frames = len(actions)

if frame_key is not None:
    frame_indices = np.asarray(data[frame_key]).reshape(-1)
else:
    frame_indices = np.arange(n_frames)


print("\nOriginal Train Frames:")
print(n_frames)

print("\nAction Shape:")
print(actions.shape)


# ------------------------------------------------------------
# 2. Compute Action Change
# ------------------------------------------------------------

action_change = np.full(n_frames, np.nan, dtype=np.float64)

for i in range(1, n_frames):

    # Do NOT compare across episode boundary
    if episode_indices[i] != episode_indices[i - 1]:
        continue

    action_change[i] = np.mean(
        np.abs(
            actions[i] - actions[i - 1]
        )
    )


valid_mask = ~np.isnan(action_change)

valid_changes = action_change[valid_mask]

print("\nValid Action Change Frames:")
print(len(valid_changes))


# ------------------------------------------------------------
# 3. Top 10% High-dynamics Threshold
# ------------------------------------------------------------

threshold = np.percentile(
    valid_changes,
    100 - TOP_PERCENT,
)

high_dynamic_mask = (
    valid_mask
    &
    (action_change >= threshold)
)

high_dynamic_indices = np.where(
    high_dynamic_mask
)[0]

n_high_dynamic = len(high_dynamic_indices)


print("\nHigh-dynamics Threshold:")
print(threshold)

print("\nHigh-dynamics Frames:")
print(n_high_dynamic)

print("\nHigh-dynamics Ratio:")
print(
    f"{n_high_dynamic / n_frames * 100:.2f}%"
)


# ------------------------------------------------------------
# 4. Build Oversampled Train Index
# ------------------------------------------------------------

original_indices = np.arange(n_frames)

extra_indices = np.repeat(
    high_dynamic_indices,
    EXTRA_COPIES,
)

expanded_indices = np.concatenate(
    [
        original_indices,
        extra_indices,
    ]
)

new_n_frames = len(expanded_indices)


print("\nExtra Samples:")
print(len(extra_indices))

print("\nDataset v2 Train Frames:")
print(new_n_frames)


# ------------------------------------------------------------
# 5. Construct Dataset v2
# ------------------------------------------------------------

output_arrays = {}

for key in data.files:

    arr = data[key]

    # Frame-level field
    if (
        hasattr(arr, "shape")
        and arr.ndim > 0
        and arr.shape[0] == n_frames
    ):
        output_arrays[key] = arr[expanded_indices]

    # Metadata / scalar field
    else:
        output_arrays[key] = arr


output_train_path = OUTPUT_DIR / "train.npz"

np.savez_compressed(
    output_train_path,
    **output_arrays,
)


# ------------------------------------------------------------
# 6. Keep Val / Test unchanged
# ------------------------------------------------------------

shutil.copy2(
    VAL_PATH,
    OUTPUT_DIR / "val.npz",
)

shutil.copy2(
    TEST_PATH,
    OUTPUT_DIR / "test.npz",
)


# ------------------------------------------------------------
# 7. Save High-dynamics Sample Report
# ------------------------------------------------------------

detail_report_path = (
    REPORT_DIR
    / "dataset_v2_high_dynamics.csv"
)

with open(
    detail_report_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "source_index",
            "episode_index",
            "frame_index",
            "action_change",
            "extra_copies",
        ]
    )

    for idx in high_dynamic_indices:

        writer.writerow(
            [
                int(idx),
                int(episode_indices[idx]),
                int(frame_indices[idx]),
                float(action_change[idx]),
                EXTRA_COPIES,
            ]
        )


# ------------------------------------------------------------
# 8. Save Summary Report
# ------------------------------------------------------------

summary_report_path = (
    REPORT_DIR
    / "dataset_v2_build_summary.csv"
)

summary_rows = [
    ["metric", "value"],

    [
        "dataset_v1_train_frames",
        n_frames,
    ],

    [
        "action_change_top_percent",
        TOP_PERCENT,
    ],

    [
        "action_change_threshold",
        threshold,
    ],

    [
        "high_dynamic_frames",
        n_high_dynamic,
    ],

    [
        "high_dynamic_ratio_percent",
        n_high_dynamic / n_frames * 100,
    ],

    [
        "extra_copies_per_high_dynamic_frame",
        EXTRA_COPIES,
    ],

    [
        "added_training_samples",
        len(extra_indices),
    ],

    [
        "dataset_v2_train_frames",
        new_n_frames,
    ],

    [
        "dataset_v1_val_frames",
        len(np.load(VAL_PATH, allow_pickle=True)[action_key]),
    ],

    [
        "dataset_v1_test_frames",
        len(np.load(TEST_PATH, allow_pickle=True)[action_key]),
    ],
]


with open(
    summary_report_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)
    writer.writerows(summary_rows)


print("\n" + "=" * 70)
print("Dataset v2 Construction Complete")
print("=" * 70)

print("\nOutput:")
print(output_train_path)

print("\nReports:")
print(summary_report_path)
print(detail_report_path)

print("\nValidation/Test:")
print("Copied unchanged from Dataset v1")