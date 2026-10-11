import csv
import os
import numpy as np


# =========================================================
# 1. Paths
# =========================================================

test_path = "dataset_v1/test.npz"

v1_prediction_path = (
    "reports/baseline_test_predictions.npz"
)

output_path = (
    "reports/per_joint_dynamics_error_analysis.csv"
)


# =========================================================
# 2. Load Data
# =========================================================

test_data = np.load(test_path)
prediction_data = np.load(v1_prediction_path)

action = test_data["action"]
episode_index = test_data["episode_index"]

absolute_error = prediction_data["absolute_error"]

assert action.shape == (2000, 14)
assert absolute_error.shape == (2000, 14)

print("Data Loading Check: PASS ✅")


# =========================================================
# 3. Compute Per-joint Action Change
#
# 对每个 Joint：
#
# |Action_j(t) - Action_j(t-1)|
#
# Episode 第一帧不计算
# =========================================================

n_frames = len(action)
n_dims = action.shape[1]

joint_action_change = np.full(
    (n_frames, n_dims),
    np.nan,
    dtype=np.float64
)

for i in range(1, n_frames):

    # 不允许跨 Episode 计算
    if episode_index[i] != episode_index[i - 1]:
        continue

    joint_action_change[i] = np.abs(
        action[i] - action[i - 1]
    )


valid_mask = np.isfinite(
    joint_action_change[:, 0]
)

print(
    "Valid Dynamic Frames:",
    int(np.sum(valid_mask))
)

assert np.sum(valid_mask) == 1995

print("Episode Boundary Check: PASS ✅")


# =========================================================
# 4. Joint Names
# =========================================================

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


# =========================================================
# 5. Per-joint Correlation
#
# Corr(
#     Joint Action Change,
#     Same Joint Prediction Error
# )
# =========================================================

results = []

print()
print("=" * 80)
print("Per-joint Dynamics × Error Correlation")
print("=" * 80)

for dim in range(n_dims):

    dynamic = joint_action_change[
        valid_mask,
        dim
    ]

    error = absolute_error[
        valid_mask,
        dim
    ]

    correlation = float(
        np.corrcoef(
            dynamic,
            error
        )[0, 1]
    )

    mean_dynamic = float(
        np.mean(dynamic)
    )

    mean_error = float(
        np.mean(error)
    )

    # -----------------------------------------------------
    # High-dynamics Top 10%
    # -----------------------------------------------------

    threshold = np.percentile(
        dynamic,
        90
    )

    high_mask = (
        dynamic >= threshold
    )

    normal_mask = (
        dynamic < threshold
    )

    high_error = float(
        np.mean(
            error[high_mask]
        )
    )

    normal_error = float(
        np.mean(
            error[normal_mask]
        )
    )

    if normal_error > 0:

        high_vs_normal_ratio = (
            high_error / normal_error
        )

    else:

        high_vs_normal_ratio = np.nan


    result = {
        "dim": dim,
        "joint": joint_names[dim],
        "correlation": correlation,
        "mean_action_change": mean_dynamic,
        "mean_absolute_error": mean_error,
        "top10_threshold": float(threshold),
        "high_dynamic_error": high_error,
        "normal_dynamic_error": normal_error,
        "high_vs_normal_error_ratio": float(
            high_vs_normal_ratio
        ),
    }

    results.append(result)

    print(
        f"Dim {dim:02d} | "
        f"{joint_names[dim]:22s} | "
        f"Corr: {correlation:+.4f} | "
        f"High/Normal Error: "
        f"{high_vs_normal_ratio:.3f}x"
    )


# =========================================================
# 6. Ranking by Correlation
# =========================================================

sorted_results = sorted(
    results,
    key=lambda x: x["correlation"],
    reverse=True
)

print()
print("=" * 80)
print("Correlation Ranking")
print("=" * 80)

for rank, item in enumerate(
    sorted_results,
    start=1
):

    print(
        f"{rank:02d}. "
        f"Dim {item['dim']:02d} | "
        f"{item['joint']:22s} | "
        f"Correlation: "
        f"{item['correlation']:+.4f}"
    )


# =========================================================
# 7. Highlight Dim 12
# =========================================================

dim12 = results[12]

print()
print("=" * 80)
print("Dim 12 Focus: right_wrist_rotate")
print("=" * 80)

print(
    "Correlation:",
    dim12["correlation"]
)

print(
    "Mean Action Change:",
    dim12["mean_action_change"]
)

print(
    "Mean Absolute Error:",
    dim12["mean_absolute_error"]
)

print(
    "Top 10% Dynamics Threshold:",
    dim12["top10_threshold"]
)

print(
    "High-dynamics Error:",
    dim12["high_dynamic_error"]
)

print(
    "Normal-dynamics Error:",
    dim12["normal_dynamic_error"]
)

print(
    "High / Normal Error Ratio:",
    dim12["high_vs_normal_error_ratio"]
)


# =========================================================
# 8. Save Report
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

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dim",
            "joint",
            "correlation",
            "mean_action_change",
            "mean_absolute_error",
            "top10_threshold",
            "high_dynamic_error",
            "normal_dynamic_error",
            "high_vs_normal_error_ratio",
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print()
print("Report:")
print(output_path)

print()
print(
    "Per-joint Dynamics Analysis: PASS ✅"
)