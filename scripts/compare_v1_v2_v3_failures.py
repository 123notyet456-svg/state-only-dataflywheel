import csv
import os
import numpy as np


# =========================================================
# 1. Paths
# =========================================================

test_path = "dataset_v1/test.npz"

v1_path = (
    "reports/baseline_test_predictions.npz"
)

v2_path = (
    "reports/v2_test_predictions.npz"
)

v3_path = (
    "reports/v3_test_predictions.npz"
)

output_path = (
    "reports/v1_v2_v3_failure_comparison.csv"
)


# =========================================================
# 2. Load Data
# =========================================================

test_data = np.load(test_path)

v1 = np.load(v1_path)
v2 = np.load(v2_path)
v3 = np.load(v3_path)


action = test_data["action"]

episode_index = (
    test_data["episode_index"]
)

frame_index = (
    test_data["frame_index"]
)


v1_error = v1["absolute_error"]
v2_error = v2["absolute_error"]
v3_error = v3["absolute_error"]


assert v1_error.shape == (2000, 14)
assert v2_error.shape == (2000, 14)
assert v3_error.shape == (2000, 14)


# =========================================================
# 3. Ground Truth Consistency
# =========================================================

assert np.array_equal(
    v1["ground_truth_action"],
    v2["ground_truth_action"]
)

assert np.array_equal(
    v1["ground_truth_action"],
    v3["ground_truth_action"]
)


print(
    "Ground Truth Consistency: PASS ✅"
)


# =========================================================
# 4. Frame-level Mean MAE
# =========================================================

v1_frame_mae = np.mean(
    v1_error,
    axis=1
)

v2_frame_mae = np.mean(
    v2_error,
    axis=1
)

v3_frame_mae = np.mean(
    v3_error,
    axis=1
)


# =========================================================
# 5. Overall Action Dynamics
# =========================================================

overall_action_change = np.full(
    len(action),
    np.nan
)


for i in range(
    1,
    len(action)
):

    if (
        episode_index[i]
        !=
        episode_index[i - 1]
    ):
        continue

    overall_action_change[i] = np.mean(
        np.abs(
            action[i]
            -
            action[i - 1]
        )
    )


valid_mask = np.isfinite(
    overall_action_change
)


overall_threshold = np.percentile(
    overall_action_change[
        valid_mask
    ],
    90
)


overall_high_mask = (
    valid_mask
    &
    (
        overall_action_change
        >=
        overall_threshold
    )
)


# =========================================================
# 6. Dim12-specific Dynamics
# =========================================================

TARGET_DIM = 12


dim12_action_change = np.full(
    len(action),
    np.nan
)


for i in range(
    1,
    len(action)
):

    if (
        episode_index[i]
        !=
        episode_index[i - 1]
    ):
        continue

    dim12_action_change[i] = abs(
        action[i, TARGET_DIM]
        -
        action[i - 1, TARGET_DIM]
    )


dim12_valid_mask = np.isfinite(
    dim12_action_change
)


dim12_threshold = np.percentile(
    dim12_action_change[
        dim12_valid_mask
    ],
    90
)


dim12_high_mask = (
    dim12_valid_mask
    &
    (
        dim12_action_change
        >=
        dim12_threshold
    )
)


dim12_normal_mask = (
    dim12_valid_mask
    &
    (~dim12_high_mask)
)


# =========================================================
# 7. Target Region
# =========================================================

target_region_mask = (
    (episode_index == 49)
    &
    (frame_index >= 292)
    &
    (frame_index <= 300)
)


frame294_mask = (
    (episode_index == 49)
    &
    (frame_index == 294)
)


# =========================================================
# 8. Helper Functions
# =========================================================

def mean_frame_error(
    frame_mae,
    mask
):

    return float(
        np.mean(
            frame_mae[mask]
        )
    )


def mean_joint_error(
    error,
    mask,
    dim
):

    return float(
        np.mean(
            error[
                mask,
                dim
            ]
        )
    )


def pct_change(
    baseline,
    value
):

    return (
        (value - baseline)
        /
        baseline
        *
        100
    )


# =========================================================
# 9. Metrics
# =========================================================

metrics = {}


# ---------------------------------------------------------
# Overall high-dynamics frame MAE
# ---------------------------------------------------------

metrics[
    "overall_high_dynamic_mae"
] = [

    mean_frame_error(
        v1_frame_mae,
        overall_high_mask
    ),

    mean_frame_error(
        v2_frame_mae,
        overall_high_mask
    ),

    mean_frame_error(
        v3_frame_mae,
        overall_high_mask
    ),
]


# ---------------------------------------------------------
# Dim12 High-dynamics Error
# ---------------------------------------------------------

metrics[
    "dim12_high_dynamic_error"
] = [

    mean_joint_error(
        v1_error,
        dim12_high_mask,
        TARGET_DIM
    ),

    mean_joint_error(
        v2_error,
        dim12_high_mask,
        TARGET_DIM
    ),

    mean_joint_error(
        v3_error,
        dim12_high_mask,
        TARGET_DIM
    ),
]


# ---------------------------------------------------------
# Dim12 Normal-dynamics Error
# ---------------------------------------------------------

metrics[
    "dim12_normal_dynamic_error"
] = [

    mean_joint_error(
        v1_error,
        dim12_normal_mask,
        TARGET_DIM
    ),

    mean_joint_error(
        v2_error,
        dim12_normal_mask,
        TARGET_DIM
    ),

    mean_joint_error(
        v3_error,
        dim12_normal_mask,
        TARGET_DIM
    ),
]


# ---------------------------------------------------------
# Episode49 Frames292-300 overall MAE
# ---------------------------------------------------------

metrics[
    "episode49_frames292_300_mae"
] = [

    mean_frame_error(
        v1_frame_mae,
        target_region_mask
    ),

    mean_frame_error(
        v2_frame_mae,
        target_region_mask
    ),

    mean_frame_error(
        v3_frame_mae,
        target_region_mask
    ),
]


# ---------------------------------------------------------
# Frame294 overall MAE
# ---------------------------------------------------------

metrics[
    "episode49_frame294_mae"
] = [

    mean_frame_error(
        v1_frame_mae,
        frame294_mask
    ),

    mean_frame_error(
        v2_frame_mae,
        frame294_mask
    ),

    mean_frame_error(
        v3_frame_mae,
        frame294_mask
    ),
]


# ---------------------------------------------------------
# Dim12 Error in Frames292-300
# ---------------------------------------------------------

metrics[
    "episode49_frames292_300_dim12_mae"
] = [

    mean_joint_error(
        v1_error,
        target_region_mask,
        TARGET_DIM
    ),

    mean_joint_error(
        v2_error,
        target_region_mask,
        TARGET_DIM
    ),

    mean_joint_error(
        v3_error,
        target_region_mask,
        TARGET_DIM
    ),
]


# =========================================================
# 10. Print
# =========================================================

print()
print("=" * 80)
print(
    "v1 vs v2 vs v3 Target Failure Comparison"
)
print("=" * 80)


for metric, values in metrics.items():

    v1_value = values[0]
    v2_value = values[1]
    v3_value = values[2]

    print()
    print(metric)

    print(
        "v1:",
        v1_value
    )

    print(
        "v2:",
        v2_value,
        "| vs v1:",
        f"{pct_change(v1_value, v2_value):+.2f}%"
    )

    print(
        "v3:",
        v3_value,
        "| vs v1:",
        f"{pct_change(v1_value, v3_value):+.2f}%"
    )

    print(
        "v3 vs v2:",
        f"{pct_change(v2_value, v3_value):+.2f}%"
    )


# =========================================================
# 11. Save Report
# =========================================================

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

    writer.writerow([
        "metric",
        "v1",
        "v2",
        "v3",
        "v2_vs_v1_percent",
        "v3_vs_v1_percent",
        "v3_vs_v2_percent",
    ])


    for metric, values in metrics.items():

        v1_value = values[0]
        v2_value = values[1]
        v3_value = values[2]

        writer.writerow([
            metric,
            v1_value,
            v2_value,
            v3_value,

            pct_change(
                v1_value,
                v2_value
            ),

            pct_change(
                v1_value,
                v3_value
            ),

            pct_change(
                v2_value,
                v3_value
            ),
        ])


    writer.writerow([
        "overall_action_change_threshold",
        overall_threshold,
        "",
        "",
        "",
        "",
        "",
    ])


    writer.writerow([
        "dim12_action_change_threshold",
        dim12_threshold,
        "",
        "",
        "",
        "",
        "",
    ])


# =========================================================
# 12. Final Summary
# =========================================================

print()

print(
    "Overall High-dynamics Frames:",
    int(
        np.sum(
            overall_high_mask
        )
    )
)

print(
    "Dim12 High-dynamics Frames:",
    int(
        np.sum(
            dim12_high_mask
        )
    )
)


print()
print(
    "Report:",
    output_path
)

print()
print(
    "v1/v2/v3 Failure Comparison: PASS ✅"
)