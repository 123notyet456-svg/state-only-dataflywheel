import os
import csv
import numpy as np


# =========================================================
# 1. Paths
# =========================================================

test_path = "dataset_v1/test.npz"

v1_prediction_path = (
    "reports/baseline_test_predictions.npz"
)

v2_prediction_path = (
    "reports/v2_test_predictions.npz"
)

output_path = (
    "reports/v1_v2_failure_comparison.csv"
)


# =========================================================
# 2. Load Data
# =========================================================

test_data = np.load(test_path)

v1 = np.load(v1_prediction_path)
v2 = np.load(v2_prediction_path)


episode_index = test_data["episode_index"]
frame_index = test_data["frame_index"]
action = test_data["action"]


v1_abs_error = v1["absolute_error"]
v2_abs_error = v2["absolute_error"]


assert v1_abs_error.shape == (2000, 14)
assert v2_abs_error.shape == (2000, 14)


# =========================================================
# 3. Check Same Ground Truth
# =========================================================

assert np.array_equal(
    v1["ground_truth_action"],
    v2["ground_truth_action"]
)

print("Ground Truth Consistency: PASS ✅")


# =========================================================
# 4. Frame Mean MAE
# =========================================================

v1_frame_mae = np.mean(
    v1_abs_error,
    axis=1
)

v2_frame_mae = np.mean(
    v2_abs_error,
    axis=1
)


# =========================================================
# 5. Compute Test Action Change
# =========================================================

action_change = np.full(
    len(action),
    np.nan
)

for i in range(1, len(action)):

    if (
        episode_index[i]
        != episode_index[i - 1]
    ):
        continue

    action_change[i] = np.mean(
        np.abs(
            action[i]
            - action[i - 1]
        )
    )


valid_mask = np.isfinite(
    action_change
)


# =========================================================
# 6. Define Test High-dynamics Top 10%
# =========================================================

threshold = np.percentile(
    action_change[valid_mask],
    90
)

high_dynamic_mask = (
    valid_mask
    &
    (action_change >= threshold)
)

normal_dynamic_mask = (
    valid_mask
    &
    (~high_dynamic_mask)
)


print()
print("===== Test Dynamics =====")

print(
    "Action Change Threshold:",
    threshold
)

print(
    "High-dynamics Frames:",
    np.sum(high_dynamic_mask)
)


# =========================================================
# 7. High-dynamics Error Comparison
# =========================================================

v1_high_mae = float(
    np.mean(
        v1_frame_mae[
            high_dynamic_mask
        ]
    )
)

v2_high_mae = float(
    np.mean(
        v2_frame_mae[
            high_dynamic_mask
        ]
    )
)


v1_normal_mae = float(
    np.mean(
        v1_frame_mae[
            normal_dynamic_mask
        ]
    )
)

v2_normal_mae = float(
    np.mean(
        v2_frame_mae[
            normal_dynamic_mask
        ]
    )
)


# =========================================================
# 8. Episode 49 Frames 292–300
# =========================================================

target_region_mask = (
    (episode_index == 49)
    &
    (frame_index >= 292)
    &
    (frame_index <= 300)
)


v1_region_mae = float(
    np.mean(
        v1_frame_mae[
            target_region_mask
        ]
    )
)

v2_region_mae = float(
    np.mean(
        v2_frame_mae[
            target_region_mask
        ]
    )
)


# =========================================================
# 9. Worst Frame 294
# =========================================================

frame294_mask = (
    (episode_index == 49)
    &
    (frame_index == 294)
)


v1_frame294_mae = float(
    np.mean(
        v1_frame_mae[
            frame294_mask
        ]
    )
)

v2_frame294_mae = float(
    np.mean(
        v2_frame_mae[
            frame294_mask
        ]
    )
)


# =========================================================
# 10. Dim 12 Error in Target Region
# =========================================================

dim12 = 12


v1_region_dim12 = float(
    np.mean(
        v1_abs_error[
            target_region_mask,
            dim12
        ]
    )
)

v2_region_dim12 = float(
    np.mean(
        v2_abs_error[
            target_region_mask,
            dim12
        ]
    )
)


# =========================================================
# 11. Percentage Change Function
# =========================================================

def pct_change(v1_value, v2_value):

    return (
        (v2_value - v1_value)
        / v1_value
        * 100
    )


# =========================================================
# 12. Print Comparison
# =========================================================

print()
print("=" * 70)
print("v1 vs v2 Target Failure Comparison")
print("=" * 70)


print()
print("High-dynamics Frames MAE")

print("v1:", v1_high_mae)
print("v2:", v2_high_mae)

print(
    "Change:",
    pct_change(
        v1_high_mae,
        v2_high_mae
    ),
    "%"
)


print()
print("Normal-dynamics Frames MAE")

print("v1:", v1_normal_mae)
print("v2:", v2_normal_mae)

print(
    "Change:",
    pct_change(
        v1_normal_mae,
        v2_normal_mae
    ),
    "%"
)


print()
print("Episode 49 Frames 292-300")

print("v1:", v1_region_mae)
print("v2:", v2_region_mae)

print(
    "Change:",
    pct_change(
        v1_region_mae,
        v2_region_mae
    ),
    "%"
)


print()
print("Episode 49 Frame 294")

print("v1:", v1_frame294_mae)
print("v2:", v2_frame294_mae)

print(
    "Change:",
    pct_change(
        v1_frame294_mae,
        v2_frame294_mae
    ),
    "%"
)


print()
print("Dim 12 Error in Frames 292-300")

print("v1:", v1_region_dim12)
print("v2:", v2_region_dim12)

print(
    "Change:",
    pct_change(
        v1_region_dim12,
        v2_region_dim12
    ),
    "%"
)


# =========================================================
# 13. Save Report
# =========================================================

rows = [
    [
        "metric",
        "v1",
        "v2",
        "change_percent"
    ],

    [
        "high_dynamic_mae",
        v1_high_mae,
        v2_high_mae,
        pct_change(
            v1_high_mae,
            v2_high_mae
        )
    ],

    [
        "normal_dynamic_mae",
        v1_normal_mae,
        v2_normal_mae,
        pct_change(
            v1_normal_mae,
            v2_normal_mae
        )
    ],

    [
        "episode49_frames292_300_mae",
        v1_region_mae,
        v2_region_mae,
        pct_change(
            v1_region_mae,
            v2_region_mae
        )
    ],

    [
        "episode49_frame294_mae",
        v1_frame294_mae,
        v2_frame294_mae,
        pct_change(
            v1_frame294_mae,
            v2_frame294_mae
        )
    ],

    [
        "episode49_frames292_300_dim12_mae",
        v1_region_dim12,
        v2_region_dim12,
        pct_change(
            v1_region_dim12,
            v2_region_dim12
        )
    ],

    [
        "test_high_dynamic_threshold",
        threshold,
        "",
        ""
    ],
]


os.makedirs(
    "reports",
    exist_ok=True
)

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerows(rows)


print()
print("Comparison Report:")
print(output_path)

print()
print("Target Failure Comparison: PASS ✅")