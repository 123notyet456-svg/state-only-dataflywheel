import csv
import os

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 1. File Paths
# 目的：
# 读取 Baseline Test Prediction 和 Train Action Statistics，
# 并统一定义 Failure Analysis 输出文件。
# =========================================================

prediction_path = "reports/baseline_test_predictions.npz"
action_statistics_path = "reports/action_statistics.csv"

joint_report_path = "reports/baseline_failure_per_joint.csv"
episode_report_path = "reports/baseline_failure_per_episode.csv"
frame_report_path = "reports/baseline_failure_frames.csv"
summary_report_path = "reports/baseline_failure_summary.csv"

joint_plot_path = "reports/baseline_failure_per_joint.png"
normalized_joint_plot_path = (
    "reports/baseline_failure_normalized_joint.png"
)
episode_plot_path = "reports/baseline_failure_per_episode.png"

os.makedirs("reports", exist_ok=True)


# =========================================================
# 2. Load Test Predictions
# 目的：
# 读取之前 Offline Evaluation 保存的预测结果，
# 避免重新运行模型。
# =========================================================

data = np.load(prediction_path)

episode_index = data["episode_index"]
frame_index = data["frame_index"]

ground_truth_action = data["ground_truth_action"]
predicted_action = data["predicted_action"]

absolute_error = data["absolute_error"]
squared_error = data["squared_error"]


assert ground_truth_action.shape == (2000, 14)
assert predicted_action.shape == (2000, 14)
assert absolute_error.shape == (2000, 14)

print("Prediction Data Loading Check: PASS ✅")


# =========================================================
# 3. Load Train Action Statistics
# 目的：
# 获取每个 Action Dimension 在 Train Set 中的 Std，
# 后面用于计算 Normalized MAE。
#
# 这样可以避免因为不同 Joint 数值尺度不同，
# 而直接使用 Raw MAE 得出错误结论。
# =========================================================

with open(
    action_statistics_path,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)
    stat_rows = list(reader)


assert len(stat_rows) == 14

columns = list(stat_rows[0].keys())

print()
print("Action Statistics Columns:", columns)


# 自动寻找 Std 列
std_column = None

for col in columns:
    if "std" in col.lower():
        std_column = col
        break


if std_column is None:
    raise ValueError(
        "Cannot find Std column in action_statistics.csv"
    )


# 自动寻找 Joint / Action 名称列
name_candidates = [
    "joint_name",
    "action_name",
    "joint",
    "name",
    "action_dimension",
    "dimension",
    "dim",
]

name_column = None

for candidate in name_candidates:
    for col in columns:
        if col.lower() == candidate:
            name_column = col
            break

    if name_column is not None:
        break


train_action_std = []

action_names = []

for dim, row in enumerate(stat_rows):

    train_action_std.append(
        float(row[std_column])
    )

    if name_column is not None:
        action_names.append(
            str(row[name_column])
        )
    else:
        action_names.append(
            f"Action Dim {dim:02d}"
        )


train_action_std = np.array(
    train_action_std,
    dtype=np.float64
)


assert train_action_std.shape == (14,)

print("Action Statistics Loading Check: PASS ✅")


# =========================================================
# 4. Overall Error
# 目的：
# 保存整个 Test Set 的总体误差，
# 作为 Failure Analysis 的全局参考值。
# =========================================================

overall_mae = float(
    np.mean(absolute_error)
)

overall_mse = float(
    np.mean(squared_error)
)

print()
print("Overall MAE:", overall_mae)
print("Overall MSE:", overall_mse)


# =========================================================
# 5. Per-joint Failure Analysis
# 目的：
# 对 14 个 Action Dimension 分别计算误差，
# 找到 Raw MAE 最大的 Joint。
# =========================================================

per_joint_mae = np.mean(
    absolute_error,
    axis=0
)

per_joint_mse = np.mean(
    squared_error,
    axis=0
)


# =========================================================
# 6. Normalized Joint Error
# 目的：
# 使用 Train Action Std 对 MAE 进行归一化，
# 公平比较不同数值尺度的 Joint。
#
# Normalized MAE =
# Test MAE / Train Action Std
# =========================================================

normalized_mae = np.full(
    14,
    np.nan,
    dtype=np.float64
)

valid_std = train_action_std > 1e-12

normalized_mae[valid_std] = (
    per_joint_mae[valid_std]
    / train_action_std[valid_std]
)


raw_joint_order = np.argsort(
    per_joint_mae
)[::-1]

normalized_joint_order = np.argsort(
    np.nan_to_num(
        normalized_mae,
        nan=-1
    )
)[::-1]


print()
print("===== Top Raw MAE Joints =====")

for rank, dim in enumerate(
    raw_joint_order[:5],
    start=1
):
    print(
        f"{rank}. Dim {dim:02d} | "
        f"{action_names[dim]} | "
        f"MAE={per_joint_mae[dim]:.8f}"
    )


print()
print("===== Top Normalized MAE Joints =====")

for rank, dim in enumerate(
    normalized_joint_order[:5],
    start=1
):
    print(
        f"{rank}. Dim {dim:02d} | "
        f"{action_names[dim]} | "
        f"MAE={per_joint_mae[dim]:.8f} | "
        f"Std={train_action_std[dim]:.8f} | "
        f"Normalized MAE={normalized_mae[dim]:.8f}"
    )


# =========================================================
# 7. Save Per-joint Failure Report
# 目的：
# 沉淀 Raw Error + Action Scale + Normalized Error，
# 后续 Data Gap Analysis 直接使用。
# =========================================================

with open(
    joint_report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "action_dimension",
        "action_name",
        "train_std",
        "test_mae",
        "test_mse",
        "normalized_mae",
    ])

    for dim in range(14):

        writer.writerow([
            dim,
            action_names[dim],
            train_action_std[dim],
            per_joint_mae[dim],
            per_joint_mse[dim],
            normalized_mae[dim],
        ])


# =========================================================
# 8. Frame-level Failure Analysis
# 目的：
# 对每一个 Test Frame 计算跨 14 个 Action Dimension
# 的平均误差。
#
# 然后找到：
# - Worst Frames
# - 每个 Worst Frame 中误差最大的 Joint
# =========================================================

frame_mae = np.mean(
    absolute_error,
    axis=1
)

frame_mse = np.mean(
    squared_error,
    axis=1
)

worst_dimension_per_frame = np.argmax(
    absolute_error,
    axis=1
)

worst_dimension_error = np.max(
    absolute_error,
    axis=1
)


frame_order = np.argsort(
    frame_mae
)[::-1]


print()
print("===== Top 10 Worst Frames =====")

for rank, idx in enumerate(
    frame_order[:10],
    start=1
):

    worst_dim = int(
        worst_dimension_per_frame[idx]
    )

    print(
        f"{rank}. "
        f"Episode {int(episode_index[idx])} | "
        f"Frame {int(frame_index[idx])} | "
        f"Frame MAE={frame_mae[idx]:.8f} | "
        f"Worst Dim={worst_dim:02d} "
        f"({action_names[worst_dim]}) | "
        f"Dim Error={worst_dimension_error[idx]:.8f}"
    )


# =========================================================
# 9. Save Frame Failure Report
# 目的：
# 将所有 2000 个 Test Frames 按误差从高到低排序，
# 后续可直接抽取 Top Failure Samples。
# =========================================================

with open(
    frame_report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "rank",
        "episode",
        "frame_index",
        "frame_mae",
        "frame_mse",
        "worst_action_dimension",
        "worst_action_name",
        "worst_dimension_absolute_error",
    ])

    for rank, idx in enumerate(
        frame_order,
        start=1
    ):

        worst_dim = int(
            worst_dimension_per_frame[idx]
        )

        writer.writerow([
            rank,
            int(episode_index[idx]),
            int(frame_index[idx]),
            frame_mae[idx],
            frame_mse[idx],
            worst_dim,
            action_names[worst_dim],
            worst_dimension_error[idx],
        ])


# =========================================================
# 10. Episode-level Failure Analysis
# 目的：
# 不只比较每个 Episode 的平均 MAE，
# 还分析其中最坏 Frame 和高误差尾部。
#
# P95 Frame MAE：
# 代表该 Episode 中误差最高的一小部分区域。
# =========================================================

episode_results = []

unique_episodes = np.unique(
    episode_index
)


for episode in unique_episodes:

    mask = (
        episode_index == episode
    )

    episode_frame_mae = frame_mae[mask]
    episode_frame_mse = frame_mse[mask]

    local_frame_indices = frame_index[mask]

    worst_local_index = np.argmax(
        episode_frame_mae
    )

    episode_results.append({
        "episode": int(episode),

        "frames": int(
            np.sum(mask)
        ),

        "mean_mae": float(
            np.mean(episode_frame_mae)
        ),

        "mean_mse": float(
            np.mean(episode_frame_mse)
        ),

        "p95_frame_mae": float(
            np.percentile(
                episode_frame_mae,
                95
            )
        ),

        "max_frame_mae": float(
            np.max(episode_frame_mae)
        ),

        "worst_frame_index": int(
            local_frame_indices[
                worst_local_index
            ]
        ),
    })


episode_results = sorted(
    episode_results,
    key=lambda x: x["mean_mae"],
    reverse=True
)


print()
print("===== Episode Failure Ranking =====")

for rank, result in enumerate(
    episode_results,
    start=1
):

    print(
        f"{rank}. Episode {result['episode']} | "
        f"Mean MAE={result['mean_mae']:.8f} | "
        f"P95={result['p95_frame_mae']:.8f} | "
        f"Max={result['max_frame_mae']:.8f} | "
        f"Worst Frame={result['worst_frame_index']}"
    )


# =========================================================
# 11. Save Episode Failure Report
# 目的：
# 保存 Episode 层失败排序，
# 找到模型最难处理的完整轨迹。
# =========================================================

with open(
    episode_report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "episode",
            "frames",
            "mean_mae",
            "mean_mse",
            "p95_frame_mae",
            "max_frame_mae",
            "worst_frame_index",
        ]
    )

    writer.writeheader()

    writer.writerows(
        episode_results
    )


# =========================================================
# 12. Find Key Failures
# 目的：
# 自动提取整个 Baseline 最关键的 Failure Signal。
# =========================================================

worst_raw_dim = int(
    raw_joint_order[0]
)

worst_normalized_dim = int(
    normalized_joint_order[0]
)

worst_episode = episode_results[0]

worst_frame_global_idx = int(
    frame_order[0]
)

worst_frame_dim = int(
    worst_dimension_per_frame[
        worst_frame_global_idx
    ]
)


# =========================================================
# 13. Save Failure Summary
# 目的：
# 将最重要的 Failure 结论整理成结构化报告，
# 下一步 Data Gap Analysis 直接从这里开始。
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

    writer.writerow([
        "overall_test_mae",
        overall_mae
    ])

    writer.writerow([
        "overall_test_mse",
        overall_mse
    ])

    writer.writerow([
        "worst_raw_action_dimension",
        worst_raw_dim
    ])

    writer.writerow([
        "worst_raw_action_name",
        action_names[worst_raw_dim]
    ])

    writer.writerow([
        "worst_raw_action_mae",
        per_joint_mae[worst_raw_dim]
    ])

    writer.writerow([
        "worst_normalized_action_dimension",
        worst_normalized_dim
    ])

    writer.writerow([
        "worst_normalized_action_name",
        action_names[worst_normalized_dim]
    ])

    writer.writerow([
        "worst_normalized_action_mae",
        normalized_mae[worst_normalized_dim]
    ])

    writer.writerow([
        "worst_episode",
        worst_episode["episode"]
    ])

    writer.writerow([
        "worst_episode_mean_mae",
        worst_episode["mean_mae"]
    ])

    writer.writerow([
        "worst_frame_episode",
        int(
            episode_index[
                worst_frame_global_idx
            ]
        )
    ])

    writer.writerow([
        "worst_frame_index",
        int(
            frame_index[
                worst_frame_global_idx
            ]
        )
    ])

    writer.writerow([
        "worst_frame_mae",
        frame_mae[
            worst_frame_global_idx
        ]
    ])

    writer.writerow([
        "worst_frame_action_dimension",
        worst_frame_dim
    ])

    writer.writerow([
        "worst_frame_action_name",
        action_names[
            worst_frame_dim
        ]
    ])


# =========================================================
# 14. Raw Per-joint MAE Plot
# 目的：
# 可视化 14 个 Action Dimension 的绝对误差。
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    np.arange(14),
    per_joint_mae
)

plt.xlabel(
    "Action Dimension"
)

plt.ylabel(
    "Test MAE"
)

plt.title(
    "State-only Baseline Per-joint MAE"
)

plt.xticks(
    np.arange(14)
)

plt.tight_layout()

plt.savefig(
    joint_plot_path,
    dpi=150
)

plt.close()


# =========================================================
# 15. Normalized Per-joint Error Plot
# 目的：
# 可视化考虑 Train Action Scale 后的 Joint Error，
# 更公平地判断真正的弱项。
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    np.arange(14),
    normalized_mae
)

plt.xlabel(
    "Action Dimension"
)

plt.ylabel(
    "Normalized MAE (MAE / Train Std)"
)

plt.title(
    "State-only Baseline Normalized Per-joint Error"
)

plt.xticks(
    np.arange(14)
)

plt.tight_layout()

plt.savefig(
    normalized_joint_plot_path,
    dpi=150
)

plt.close()


# =========================================================
# 16. Episode Failure Plot
# 目的：
# 可视化 Test Episode 45–49 的平均误差差异。
# =========================================================

plot_episode_ids = [
    result["episode"]
    for result in episode_results
]

plot_episode_mae = [
    result["mean_mae"]
    for result in episode_results
]

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    plot_episode_ids,
    plot_episode_mae
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Mean Frame MAE"
)

plt.title(
    "State-only Baseline Per-episode Failure"
)

plt.tight_layout()

plt.savefig(
    episode_plot_path,
    dpi=150
)

plt.close()


# =========================================================
# 17. Final Summary
# =========================================================

print()
print("===== Baseline Failure Analysis Complete =====")

print(
    "Worst Raw Joint:",
    worst_raw_dim,
    action_names[worst_raw_dim]
)

print(
    "Worst Normalized Joint:",
    worst_normalized_dim,
    action_names[worst_normalized_dim]
)

print(
    "Worst Episode:",
    worst_episode["episode"]
)

print(
    "Worst Frame:",
    "Episode",
    int(
        episode_index[
            worst_frame_global_idx
        ]
    ),
    "Frame",
    int(
        frame_index[
            worst_frame_global_idx
        ]
    )
)

print()
print("Reports:")

print(joint_report_path)
print(episode_report_path)
print(frame_report_path)
print(summary_report_path)

print()
print("Plots:")

print(joint_plot_path)
print(normalized_joint_plot_path)
print(episode_plot_path)

print()
print("Failure Analysis: PASS ✅")