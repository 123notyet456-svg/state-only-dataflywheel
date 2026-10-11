from pathlib import Path
import csv
import shutil
import numpy as np


# =========================================================
# 1. Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "dataset_v1"
OUTPUT_DIR = PROJECT_ROOT / "dataset_v3"
REPORT_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = INPUT_DIR / "train.npz"
VAL_PATH = INPUT_DIR / "val.npz"
TEST_PATH = INPUT_DIR / "test.npz"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


TARGET_DIM = 12
TARGET_JOINT = "right_wrist_rotate"

TOP_PERCENT = 10
EXTRA_COPIES = 2


print("=" * 70)
print("Dataset v3 Construction")
print("=" * 70)

print()
print("Strategy:")
print(
    f"Target Dim {TARGET_DIM}: {TARGET_JOINT}"
)
print(
    f"Exact Top {TOP_PERCENT}% Joint-specific Dynamics"
)
print(
    f"Extra Copies: {EXTRA_COPIES}"
)


# =========================================================
# 2. Load Dataset v1 Train
# =========================================================

data = np.load(
    TRAIN_PATH,
    allow_pickle=True
)

actions = np.asarray(
    data["action"]
)

episode_indices = np.asarray(
    data["episode_index"]
).reshape(-1)

frame_indices = np.asarray(
    data["frame_index"]
).reshape(-1)

n_frames = len(actions)


print()
print("Original Train Frames:")
print(n_frames)

print("Action Shape:")
print(actions.shape)


assert actions.shape == (16000, 14)
assert len(episode_indices) == n_frames
assert len(frame_indices) == n_frames


# =========================================================
# 3. Compute Dim12 Action Change
#
# |Action_dim12(t) - Action_dim12(t-1)|
#
# Episode start frames are excluded.
# =========================================================

joint_action_change = np.full(
    n_frames,
    np.nan,
    dtype=np.float64
)


for i in range(1, n_frames):

    # Do not compute change across episode boundary
    if (
        episode_indices[i]
        != episode_indices[i - 1]
    ):
        continue

    joint_action_change[i] = abs(
        actions[i, TARGET_DIM]
        -
        actions[i - 1, TARGET_DIM]
    )


valid_mask = np.isfinite(
    joint_action_change
)

valid_indices = np.where(
    valid_mask
)[0]

valid_changes = (
    joint_action_change[
        valid_mask
    ]
)


print()
print("Valid Dynamic Frames:")
print(len(valid_changes))


# 16000 frames - 40 episode starts = 15960
assert len(valid_changes) == 15960


# =========================================================
# 4. Exact Top 10% Selection
#
# Important:
#
# Do NOT use:
#
# joint_action_change >= percentile_threshold
#
# because many samples can have exactly the same value at
# the percentile boundary, causing more than 10% samples
# to be selected.
#
# Instead:
#
# sort all valid frames by Dim12 Action Change
# ↓
# select EXACTLY the largest 10%
#
# This keeps Dataset v3 sample count aligned with Dataset v2.
# =========================================================

target_count = int(
    len(valid_changes)
    * TOP_PERCENT
    / 100
)


print()
print("Exact Target Count:")
print(target_count)


# Stable sort from smallest to largest
sorted_positions = np.argsort(
    valid_changes,
    kind="stable"
)


# Take exactly the largest target_count samples
selected_positions = (
    sorted_positions[
        -target_count:
    ]
)


high_dynamic_indices = (
    valid_indices[
        selected_positions
    ]
)


n_high_dynamic = len(
    high_dynamic_indices
)


assert n_high_dynamic == target_count
assert n_high_dynamic == 1596


# Smallest selected action change becomes
# the effective selection threshold
threshold = float(
    np.min(
        joint_action_change[
            high_dynamic_indices
        ]
    )
)


print()
print("Dim12 Dynamics Threshold:")
print(threshold)

print()
print("High-dynamics Frames:")
print(n_high_dynamic)

print()
print("High-dynamics Ratio:")
print(
    f"{n_high_dynamic / n_frames * 100:.3f}%"
)


# =========================================================
# 5. Build Oversampled Index
#
# Normal Frame:
# appears 1 time
#
# Selected Dim12 High-dynamics Frame:
# original 1 + extra 2 = 3 times
# =========================================================

original_indices = np.arange(
    n_frames
)


extra_indices = np.repeat(
    high_dynamic_indices,
    EXTRA_COPIES
)


expanded_indices = np.concatenate(
    [
        original_indices,
        extra_indices,
    ]
)


new_n_frames = len(
    expanded_indices
)


print()
print("Extra Samples:")
print(len(extra_indices))

print()
print("Dataset v3 Train Frames:")
print(new_n_frames)


assert len(extra_indices) == 3192
assert new_n_frames == 19192


# =========================================================
# 6. Construct Dataset v3 Train
# =========================================================

output_arrays = {}


for key in data.files:

    arr = data[key]

    # Frame-level data
    if (
        hasattr(arr, "shape")
        and arr.ndim > 0
        and arr.shape[0] == n_frames
    ):

        output_arrays[key] = (
            arr[
                expanded_indices
            ]
        )

    # Metadata / scalar data
    else:

        output_arrays[key] = arr


output_train_path = (
    OUTPUT_DIR
    / "train.npz"
)


np.savez_compressed(
    output_train_path,
    **output_arrays,
)


# =========================================================
# 7. Keep Validation / Test Unchanged
# =========================================================

shutil.copy2(
    VAL_PATH,
    OUTPUT_DIR / "val.npz",
)

shutil.copy2(
    TEST_PATH,
    OUTPUT_DIR / "test.npz",
)


# =========================================================
# 8. Save Selected Frame Detail Report
# =========================================================

detail_report_path = (
    REPORT_DIR
    / "dataset_v3_dim12_high_dynamics.csv"
)


with open(
    detail_report_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "source_index",
            "episode_index",
            "frame_index",
            "target_dim",
            "target_joint",
            "joint_action_change",
            "extra_copies",
        ]
    )

    for idx in high_dynamic_indices:

        writer.writerow(
            [
                int(idx),
                int(
                    episode_indices[idx]
                ),
                int(
                    frame_indices[idx]
                ),
                TARGET_DIM,
                TARGET_JOINT,
                float(
                    joint_action_change[
                        idx
                    ]
                ),
                EXTRA_COPIES,
            ]
        )


# =========================================================
# 9. Calculate Effective Sampling Ratio
#
# Selected frames appear 3 times in Dataset v3.
# =========================================================

effective_high_dynamic_samples = (
    n_high_dynamic
    * (1 + EXTRA_COPIES)
)


effective_sampling_ratio = (
    effective_high_dynamic_samples
    / new_n_frames
    * 100
)


# =========================================================
# 10. Save Summary Report
# =========================================================

summary_report_path = (
    REPORT_DIR
    / "dataset_v3_build_summary.csv"
)


val_data = np.load(
    VAL_PATH,
    allow_pickle=True
)

test_data = np.load(
    TEST_PATH,
    allow_pickle=True
)


summary_rows = [

    [
        "metric",
        "value"
    ],

    [
        "strategy",
        "joint_specific_dynamic_oversampling",
    ],

    [
        "target_dim",
        TARGET_DIM,
    ],

    [
        "target_joint",
        TARGET_JOINT,
    ],

    [
        "dataset_v1_train_frames",
        n_frames,
    ],

    [
        "valid_dynamic_frames",
        len(valid_changes),
    ],

    [
        "action_change_top_percent",
        TOP_PERCENT,
    ],

    [
        "exact_target_count",
        target_count,
    ],

    [
        "dim12_action_change_threshold",
        threshold,
    ],

    [
        "high_dynamic_frames",
        n_high_dynamic,
    ],

    [
        "high_dynamic_unique_ratio_percent",
        n_high_dynamic
        / n_frames
        * 100,
    ],

    [
        "extra_copies_per_frame",
        EXTRA_COPIES,
    ],

    [
        "added_training_samples",
        len(extra_indices),
    ],

    [
        "dataset_v3_train_frames",
        new_n_frames,
    ],

    [
        "effective_high_dynamic_samples",
        effective_high_dynamic_samples,
    ],

    [
        "dataset_v3_high_dynamic_sampling_ratio_percent",
        effective_sampling_ratio,
    ],

    [
        "validation_frames",
        len(
            val_data["action"]
        ),
    ],

    [
        "test_frames",
        len(
            test_data["action"]
        ),
    ],
]


with open(
    summary_report_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerows(
        summary_rows
    )


# =========================================================
# 11. Final Checks
# =========================================================

assert (
    output_arrays["state"].shape
    ==
    (19192, 14)
)

assert (
    output_arrays["action"].shape
    ==
    (19192, 14)
)

assert (
    len(val_data["action"])
    ==
    2000
)

assert (
    len(test_data["action"])
    ==
    2000
)


# =========================================================
# 12. Final Summary
# =========================================================

print()
print("=" * 70)
print("Dataset v3 Construction Complete")
print("=" * 70)

print()
print("Strategy:")
print(
    "Dim12 right_wrist_rotate "
    "Exact Top 10% Oversampling"
)

print()
print("Original Train Frames:")
print(n_frames)

print()
print("Selected High-dynamics Frames:")
print(n_high_dynamic)

print()
print("Added Samples:")
print(len(extra_indices))

print()
print("Dataset v3 Train Frames:")
print(new_n_frames)

print()
print(
    "Effective High-dynamics Sampling Ratio:"
)

print(
    f"{effective_sampling_ratio:.2f}%"
)

print()
print("Output:")
print(output_train_path)

print()
print("Reports:")
print(summary_report_path)
print(detail_report_path)

print()
print(
    "Validation/Test copied unchanged."
)

print()
print(
    "Dataset v3 Construction: PASS ✅"
)