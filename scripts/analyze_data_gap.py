import csv
import os

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 1. File Paths
# 目的：
# 读取 Dataset v1、Test Prediction 和之前的 Failure 结果，
# 为 Data Gap Analysis 提供统一输入。
# =========================================================

train_path = "dataset_v1/train.npz"
test_path = "dataset_v1/test.npz"

prediction_path = "reports/baseline_test_predictions.npz"

frame_report_path = "reports/data_gap_frame_analysis.csv"
summary_report_path = "reports/data_gap_summary.csv"

episode49_plot_path = "reports/data_gap_episode49_timeline.png"
distance_plot_path = "reports/data_gap_error_vs_state_distance.png"
dynamics_plot_path = "reports/data_gap_error_vs_action_change.png"

os.makedirs("reports", exist_ok=True)


# =========================================================
# 2. Load Train / Test Dataset
# 目的：
# 同时拿到 Train 和 Test 的 State / Action，
# 后续判断 Worst Test Frames 在 Train Dataset 中
# 是否有相似的数据覆盖。
# =========================================================

train_data = np.load(train_path)
test_data = np.load(test_path)

train_state = train_data["state"].astype(np.float32)
train_action = train_data["action"].astype(np.float32)

test_state = test_data["state"].astype(np.float32)
test_action = test_data["action"].astype(np.float32)

episode_index = test_data["episode_index"]
frame_index = test_data["frame_index"]

assert train_state.shape == (16000, 14)
assert train_action.shape == (16000, 14)

assert test_state.shape == (2000, 14)
assert test_action.shape == (2000, 14)

print("Train / Test Dataset Loading Check: PASS ✅")


# =========================================================
# 3. Load Prediction Error
# 目的：
# 读取之前 Offline Evaluation 保存的模型预测误差。
#
# Data Gap Analysis 的核心思想：
#
# 高误差 Frame
# ↓
# 看它的数据覆盖 / 动作变化 / 局部歧义
# =========================================================

prediction_data = np.load(prediction_path)

absolute_error = prediction_data["absolute_error"]

assert absolute_error.shape == (2000, 14)

frame_mae = np.mean(
    absolute_error,
    axis=1
)

worst_dimension_per_frame = np.argmax(
    absolute_error,
    axis=1
)

print("Prediction Error Loading Check: PASS ✅")


# =========================================================
# 4. Find Worst Raw Action Dimension
# 目的：
# 找到整个 Test Set 中绝对 MAE 最大的 Action Dimension。
# 后续观察该动作维度在局部邻居中的分布。
# =========================================================

per_joint_mae = np.mean(
    absolute_error,
    axis=0
)

worst_raw_dim = int(
    np.argmax(per_joint_mae)
)

print()
print("Worst Raw Action Dimension:", worst_raw_dim)
print("MAE:", per_joint_mae[worst_raw_dim])


# =========================================================
# 5. State Normalization
# 目的：
# 不同 State Dimension 的数值尺度可能不同。
#
# 如果直接计算欧氏距离，
# 大尺度 Dimension 会主导距离。
#
# 因此使用 Train Set Mean / Std 标准化 State。
# =========================================================

state_mean = np.mean(
    train_state,
    axis=0
)

state_std = np.std(
    train_state,
    axis=0
)

safe_state_std = state_std.copy()

safe_state_std[
    safe_state_std < 1e-8
] = 1.0

train_state_normalized = (
    train_state - state_mean
) / safe_state_std

test_state_normalized = (
    test_state - state_mean
) / safe_state_std

train_state_normalized = (
    train_state_normalized.astype(np.float32)
)

test_state_normalized = (
    test_state_normalized.astype(np.float32)
)

print("State Normalization Check: PASS ✅")


# =========================================================
# 6. Nearest-neighbor State Coverage Analysis
# 目的：
# 对每一个 Test Frame：
#
# 找 Train Dataset 中最相似的 State。
#
# 如果某个 Failure Frame 距离所有 Train State 都很远，
# 说明可能存在：
#
# State Coverage Gap（状态覆盖缺口）
#
# 同时取 K=20 个最近邻，
# 后面分析相似 State 对应的 Action 是否一致。
# =========================================================

k_neighbors = 20
chunk_size = 100

nearest_distance = np.zeros(
    len(test_state),
    dtype=np.float32
)

local_action_std = np.zeros(
    len(test_state),
    dtype=np.float32
)

local_worst_dim_action_std = np.zeros(
    len(test_state),
    dtype=np.float32
)


train_squared_norm = np.sum(
    train_state_normalized ** 2,
    axis=1
)


for start in range(
    0,
    len(test_state_normalized),
    chunk_size
):

    end = min(
        start + chunk_size,
        len(test_state_normalized)
    )

    test_chunk = test_state_normalized[
        start:end
    ]

    test_squared_norm = np.sum(
        test_chunk ** 2,
        axis=1,
        keepdims=True
    )

    # 欧氏距离平方
    distance_squared = (
        test_squared_norm
        + train_squared_norm[None, :]
        - 2.0 * (
            test_chunk
            @ train_state_normalized.T
        )
    )

    distance_squared = np.maximum(
        distance_squared,
        0.0
    )

    neighbor_indices = np.argpartition(
        distance_squared,
        kth=k_neighbors - 1,
        axis=1
    )[:, :k_neighbors]

    for local_idx in range(
        end - start
    ):

        global_idx = (
            start + local_idx
        )

        neighbors = neighbor_indices[
            local_idx
        ]

        neighbor_distances = np.sqrt(
            distance_squared[
                local_idx,
                neighbors
            ]
        )

        nearest_distance[
            global_idx
        ] = np.min(
            neighbor_distances
        )

        neighbor_actions = train_action[
            neighbors
        ]

        # 14维 Action 的局部平均标准差
        local_action_std[
            global_idx
        ] = np.mean(
            np.std(
                neighbor_actions,
                axis=0
            )
        )

        # Failure 中最突出的动作维度
        local_worst_dim_action_std[
            global_idx
        ] = np.std(
            neighbor_actions[
                :,
                worst_raw_dim
            ]
        )


print("Nearest-neighbor Coverage Analysis: PASS ✅")


# =========================================================
# 7. Action Dynamics Analysis
# 目的：
# 检查模型失败是否集中在动作快速变化阶段。
#
# 对连续 Frame 计算：
#
# |Action(t) - Action(t-1)|
#
# 如果 Worst Frames 同时存在非常大的 Action Change，
# 说明 State-only 单帧模型可能难以处理动态转换阶段。
# =========================================================

action_change = np.full(
    len(test_action),
    np.nan,
    dtype=np.float32
)

worst_dim_action_change = np.full(
    len(test_action),
    np.nan,
    dtype=np.float32
)


for i in range(
    1,
    len(test_action)
):

    same_episode = (
        episode_index[i]
        == episode_index[i - 1]
    )

    continuous_frame = (
        frame_index[i]
        == frame_index[i - 1] + 1
    )

    if (
        same_episode
        and continuous_frame
    ):

        diff = np.abs(
            test_action[i]
            - test_action[i - 1]
        )

        action_change[i] = np.mean(
            diff
        )

        worst_dim_action_change[i] = (
            diff[worst_raw_dim]
        )


print("Action Dynamics Analysis: PASS ✅")


# =========================================================
# 8. Helper: Percentile Rank
# 目的：
# 判断某个 Failure Frame 的指标处于 Test Set 的什么位置。
#
# 例如：
#
# State Distance Percentile = 98%
#
# 表示它比约 98% Test Frames
# 离 Train Dataset 更远。
# =========================================================

def percentile_rank(
    values,
    value
):

    valid_values = values[
        np.isfinite(values)
    ]

    if len(valid_values) == 0:
        return np.nan

    return float(
        np.mean(
            valid_values <= value
        ) * 100.0
    )


# =========================================================
# 9. Worst Frame Analysis
# 目的：
# 对最差 Frame 分别检查：
#
# 1. State Coverage
# 2. Local Action Ambiguity
# 3. Action Dynamics
#
# 为 Data Gap 原因提供证据。
# =========================================================

worst_frame_idx = int(
    np.argmax(frame_mae)
)

worst_frame_episode = int(
    episode_index[
        worst_frame_idx
    ]
)

worst_frame_number = int(
    frame_index[
        worst_frame_idx
    ]
)

worst_frame_error = float(
    frame_mae[
        worst_frame_idx
    ]
)


worst_frame_distance = float(
    nearest_distance[
        worst_frame_idx
    ]
)

worst_frame_distance_percentile = (
    percentile_rank(
        nearest_distance,
        worst_frame_distance
    )
)


worst_frame_local_std = float(
    local_action_std[
        worst_frame_idx
    ]
)

worst_frame_local_std_percentile = (
    percentile_rank(
        local_action_std,
        worst_frame_local_std
    )
)


worst_frame_action_change = float(
    action_change[
        worst_frame_idx
    ]
)

worst_frame_action_change_percentile = (
    percentile_rank(
        action_change,
        worst_frame_action_change
    )
)


print()
print("===== Worst Frame Data Gap Analysis =====")

print(
    "Episode:",
    worst_frame_episode
)

print(
    "Frame:",
    worst_frame_number
)

print(
    "Frame MAE:",
    worst_frame_error
)

print(
    "Nearest Train State Distance:",
    worst_frame_distance
)

print(
    "State Distance Percentile:",
    worst_frame_distance_percentile
)

print(
    "Local Action Std:",
    worst_frame_local_std
)

print(
    "Local Action Std Percentile:",
    worst_frame_local_std_percentile
)

print(
    "Action Change:",
    worst_frame_action_change
)

print(
    "Action Change Percentile:",
    worst_frame_action_change_percentile
)


# =========================================================
# 10. High-error Group Analysis
# 目的：
# 单独看一个 Frame 可能存在偶然性。
#
# 因此把 Frame MAE 最高的 5% 样本作为 High-error Group，
# 和整个 Test Set 比较。
#
# 判断：
#
# 高误差区域是否整体具有
# 更差 State Coverage /
# 更高 Action Dynamics /
# 更强 Local Action Ambiguity。
# =========================================================

error_threshold = np.percentile(
    frame_mae,
    95
)

high_error_mask = (
    frame_mae >= error_threshold
)

normal_error_mask = (
    frame_mae < error_threshold
)


def safe_mean(values, mask):

    selected = values[mask]

    selected = selected[
        np.isfinite(selected)
    ]

    if len(selected) == 0:
        return np.nan

    return float(
        np.mean(selected)
    )


high_error_distance = safe_mean(
    nearest_distance,
    high_error_mask
)

normal_error_distance = safe_mean(
    nearest_distance,
    normal_error_mask
)


high_error_local_std = safe_mean(
    local_action_std,
    high_error_mask
)

normal_error_local_std = safe_mean(
    local_action_std,
    normal_error_mask
)


high_error_action_change = safe_mean(
    action_change,
    high_error_mask
)

normal_error_action_change = safe_mean(
    action_change,
    normal_error_mask
)


print()
print("===== High-error Group vs Other Frames =====")

print("Top 5% Error Threshold:", error_threshold)

print()
print(
    "Nearest State Distance"
)

print(
    "High-error:",
    high_error_distance
)

print(
    "Others:",
    normal_error_distance
)

print()
print(
    "Local Action Std"
)

print(
    "High-error:",
    high_error_local_std
)

print(
    "Others:",
    normal_error_local_std
)

print()
print(
    "Action Change"
)

print(
    "High-error:",
    high_error_action_change
)

print(
    "Others:",
    normal_error_action_change
)


# =========================================================
# 11. Correlation Analysis
# 目的：
# 查看模型误差是否和以下因素整体相关：
#
# State Coverage Distance
# Local Action Ambiguity
# Action Dynamics
#
# 注意：
# Correlation 只是线索，不等于因果关系。
# =========================================================

coverage_correlation = float(
    np.corrcoef(
        frame_mae,
        nearest_distance
    )[0, 1]
)

ambiguity_correlation = float(
    np.corrcoef(
        frame_mae,
        local_action_std
    )[0, 1]
)


valid_dynamic_mask = np.isfinite(
    action_change
)

dynamics_correlation = float(
    np.corrcoef(
        frame_mae[
            valid_dynamic_mask
        ],
        action_change[
            valid_dynamic_mask
        ]
    )[0, 1]
)


print()
print("===== Error Correlation =====")

print(
    "Error vs State Distance:",
    coverage_correlation
)

print(
    "Error vs Local Action Std:",
    ambiguity_correlation
)

print(
    "Error vs Action Change:",
    dynamics_correlation
)


# =========================================================
# 12. Save Frame-level Data Gap Report
# 目的：
# 每个 Test Frame 保存：
#
# Error
# Coverage
# Ambiguity
# Dynamics
#
# 后续可以直接筛选高价值 Dataset v2 候选区域。
# =========================================================

frame_order = np.argsort(
    frame_mae
)[::-1]


with open(
    frame_report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "error_rank",
        "episode",
        "frame_index",
        "frame_mae",
        "nearest_train_state_distance",
        "local_action_std",
        "local_worst_dim_action_std",
        "action_change",
        "worst_dim_action_change",
        "worst_error_dimension",
    ])

    for rank, idx in enumerate(
        frame_order,
        start=1
    ):

        writer.writerow([
            rank,
            int(episode_index[idx]),
            int(frame_index[idx]),
            float(frame_mae[idx]),
            float(nearest_distance[idx]),
            float(local_action_std[idx]),
            float(local_worst_dim_action_std[idx]),
            float(action_change[idx]),
            float(worst_dim_action_change[idx]),
            int(
                worst_dimension_per_frame[idx]
            ),
        ])


# =========================================================
# 13. Save Data Gap Summary
# 目的：
# 将核心证据结构化保存，
# 下一步 Dataset v2 Design 直接使用。
# =========================================================

with open(
    summary_report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "metric",
        "value"
    ])

    summary_rows = [

        (
            "worst_frame_episode",
            worst_frame_episode
        ),

        (
            "worst_frame_index",
            worst_frame_number
        ),

        (
            "worst_frame_mae",
            worst_frame_error
        ),

        (
            "worst_frame_state_distance",
            worst_frame_distance
        ),

        (
            "worst_frame_state_distance_percentile",
            worst_frame_distance_percentile
        ),

        (
            "worst_frame_local_action_std",
            worst_frame_local_std
        ),

        (
            "worst_frame_local_action_std_percentile",
            worst_frame_local_std_percentile
        ),

        (
            "worst_frame_action_change",
            worst_frame_action_change
        ),

        (
            "worst_frame_action_change_percentile",
            worst_frame_action_change_percentile
        ),

        (
            "high_error_mean_state_distance",
            high_error_distance
        ),

        (
            "other_frames_mean_state_distance",
            normal_error_distance
        ),

        (
            "high_error_mean_local_action_std",
            high_error_local_std
        ),

        (
            "other_frames_mean_local_action_std",
            normal_error_local_std
        ),

        (
            "high_error_mean_action_change",
            high_error_action_change
        ),

        (
            "other_frames_mean_action_change",
            normal_error_action_change
        ),

        (
            "error_state_distance_correlation",
            coverage_correlation
        ),

        (
            "error_local_action_std_correlation",
            ambiguity_correlation
        ),

        (
            "error_action_change_correlation",
            dynamics_correlation
        ),
    ]

    writer.writerows(
        summary_rows
    )


# =========================================================
# 14. Episode 49 Failure Timeline
# 目的：
# 可视化 Episode 49 全 400 Frames 的模型误差。
#
# 检查 Frame 292–300 是否形成明显局部 Failure Peak。
# =========================================================

episode49_mask = (
    episode_index == 49
)

episode49_frames = frame_index[
    episode49_mask
]

episode49_error = frame_mae[
    episode49_mask
]


plt.figure(
    figsize=(11, 5)
)

plt.plot(
    episode49_frames,
    episode49_error
)

plt.xlabel(
    "Frame Index"
)

plt.ylabel(
    "Frame MAE"
)

plt.title(
    "Episode 49 Failure Timeline"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    episode49_plot_path,
    dpi=150
)

plt.close()


# =========================================================
# 15. Error vs State Coverage Plot
# 目的：
# 看 State 离 Train Dataset 越远时，
# 模型误差是否倾向更高。
# =========================================================

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    nearest_distance,
    frame_mae,
    s=10,
    alpha=0.5
)

plt.xlabel(
    "Nearest Train State Distance"
)

plt.ylabel(
    "Frame MAE"
)

plt.title(
    "Prediction Error vs State Coverage"
)

plt.tight_layout()

plt.savefig(
    distance_plot_path,
    dpi=150
)

plt.close()


# =========================================================
# 16. Error vs Action Dynamics Plot
# 目的：
# 判断动作变化越快时，
# State-only 模型是否更容易出现预测错误。
# =========================================================

dynamic_mask = np.isfinite(
    action_change
)

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    action_change[
        dynamic_mask
    ],
    frame_mae[
        dynamic_mask
    ],
    s=10,
    alpha=0.5
)

plt.xlabel(
    "Action Change"
)

plt.ylabel(
    "Frame MAE"
)

plt.title(
    "Prediction Error vs Action Dynamics"
)

plt.tight_layout()

plt.savefig(
    dynamics_plot_path,
    dpi=150
)

plt.close()


# =========================================================
# 17. Final Summary
# =========================================================

print()
print("===== Data Gap Analysis Complete =====")

print(
    "Worst Failure:",
    f"Episode {worst_frame_episode}",
    f"Frame {worst_frame_number}"
)

print()
print("Reports:")
print(frame_report_path)
print(summary_report_path)

print()
print("Plots:")
print(episode49_plot_path)
print(distance_plot_path)
print(dynamics_plot_path)

print()
print("Data Gap Analysis: PASS ✅")