import os
import csv

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# 1. Model Definition
# =========================================================

class StatePolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 14),
        )

    def forward(self, state):
        return self.net(state)


# =========================================================
# 2. File Paths
# =========================================================

test_path = "dataset_v3/test.npz"

model_path = "models/state_policy_v3_best.pt"

overall_report_path = (
    "reports/v3_test_metrics.csv"
)

joint_report_path = (
    "reports/v3_per_joint_metrics.csv"
)

episode_report_path = (
    "reports/v3_per_episode_metrics.csv"
)

prediction_path = (
    "reports/v3_test_predictions.npz"
)


# =========================================================
# 3. Load Test Dataset
# =========================================================

test_data = np.load(test_path)

test_state = torch.tensor(
    test_data["state"],
    dtype=torch.float32
)

test_action = torch.tensor(
    test_data["action"],
    dtype=torch.float32
)

episode_index = test_data["episode_index"]
frame_index = test_data["frame_index"]


print(
    "Test state shape:",
    test_state.shape
)

print(
    "Test action shape:",
    test_action.shape
)

print(
    "Test episodes:",
    np.unique(episode_index)
)


assert test_state.shape == (
    2000,
    14
)

assert test_action.shape == (
    2000,
    14
)

assert set(
    np.unique(
        episode_index
    )
) == {
    45,
    46,
    47,
    48,
    49
}


print(
    "Test Dataset Loading Check: PASS ✅"
)


# =========================================================
# 4. Load Dataset v3 Best Model
# =========================================================

model = StatePolicy()

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu"
    )
)

model.eval()


print(
    "Dataset v3 Best Model Loading Check: PASS ✅"
)


# =========================================================
# 5. Forward Pass
# =========================================================

with torch.no_grad():

    predicted_action = model(
        test_state
    )


assert predicted_action.shape == (
    2000,
    14
)


print(
    "Test prediction shape:",
    predicted_action.shape
)

print(
    "Test Forward Pass Check: PASS ✅"
)


# =========================================================
# 6. Convert to NumPy
# =========================================================

predicted_action_np = (
    predicted_action
    .cpu()
    .numpy()
)

test_action_np = (
    test_action
    .cpu()
    .numpy()
)


# =========================================================
# 7. Error Matrix
# =========================================================

absolute_error = np.abs(
    predicted_action_np
    -
    test_action_np
)

squared_error = (
    predicted_action_np
    -
    test_action_np
) ** 2


# =========================================================
# 8. Overall Metrics
# =========================================================

overall_mae = float(
    np.mean(
        absolute_error
    )
)

overall_mse = float(
    np.mean(
        squared_error
    )
)


print()
print(
    "===== Dataset v3 Overall Test Metrics ====="
)

print(
    "Test MAE:",
    overall_mae
)

print(
    "Test MSE:",
    overall_mse
)


assert np.isfinite(
    overall_mae
)

assert np.isfinite(
    overall_mse
)


print(
    "Overall Test Metrics Check: PASS ✅"
)


# =========================================================
# 9. Per-joint Metrics
# =========================================================

per_joint_mae = np.mean(
    absolute_error,
    axis=0
)

per_joint_mse = np.mean(
    squared_error,
    axis=0
)


assert per_joint_mae.shape == (14,)
assert per_joint_mse.shape == (14,)


print()
print(
    "===== Dataset v3 Per-joint Metrics ====="
)


for dim in range(14):

    print(
        f"Action Dim {dim:02d} | "
        f"MAE: {per_joint_mae[dim]:.8f} | "
        f"MSE: {per_joint_mse[dim]:.8f}"
    )


print(
    "Per-joint Metrics Check: PASS ✅"
)


# =========================================================
# 10. Per-episode Metrics
# =========================================================

episode_results = []

unique_episodes = np.unique(
    episode_index
)


print()
print(
    "===== Dataset v3 Per-episode Metrics ====="
)


for episode in unique_episodes:

    mask = (
        episode_index
        ==
        episode
    )


    episode_mae = float(
        np.mean(
            absolute_error[
                mask
            ]
        )
    )


    episode_mse = float(
        np.mean(
            squared_error[
                mask
            ]
        )
    )


    episode_frames = int(
        np.sum(
            mask
        )
    )


    episode_results.append(
        {
            "episode":
                int(episode),

            "frames":
                episode_frames,

            "mae":
                episode_mae,

            "mse":
                episode_mse,
        }
    )


    print(
        f"Episode "
        f"{int(episode):02d} | "
        f"Frames: "
        f"{episode_frames} | "
        f"MAE: "
        f"{episode_mae:.8f} | "
        f"MSE: "
        f"{episode_mse:.8f}"
    )


assert len(
    episode_results
) == 5


print(
    "Per-episode Metrics Check: PASS ✅"
)


# =========================================================
# 11. Worst Episode
# =========================================================

worst_episode = max(
    episode_results,
    key=lambda x:
        x["mae"]
)


print()
print(
    "===== Dataset v3 Worst Episode ====="
)

print(
    "Episode:",
    worst_episode[
        "episode"
    ]
)

print(
    "MAE:",
    worst_episode[
        "mae"
    ]
)

print(
    "MSE:",
    worst_episode[
        "mse"
    ]
)


# =========================================================
# 12. Save Overall Report
# =========================================================

os.makedirs(
    "reports",
    exist_ok=True
)


with open(
    overall_report_path,
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
        "test_frames",
        len(test_state)
    ])

    writer.writerow([
        "test_episodes",
        len(unique_episodes)
    ])

    writer.writerow([
        "overall_mae",
        overall_mae
    ])

    writer.writerow([
        "overall_mse",
        overall_mse
    ])

    writer.writerow([
        "worst_episode",
        worst_episode[
            "episode"
        ]
    ])

    writer.writerow([
        "worst_episode_mae",
        worst_episode[
            "mae"
        ]
    ])


# =========================================================
# 13. Save Per-joint Report
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
        "mae",
        "mse"
    ])

    for dim in range(14):

        writer.writerow([
            dim,
            per_joint_mae[dim],
            per_joint_mse[dim]
        ])


# =========================================================
# 14. Save Per-episode Report
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
            "mae",
            "mse"
        ]
    )

    writer.writeheader()

    writer.writerows(
        episode_results
    )


# =========================================================
# 15. Save Predictions
# =========================================================

np.savez(
    prediction_path,

    episode_index=
        episode_index,

    frame_index=
        frame_index,

    ground_truth_action=
        test_action_np,

    predicted_action=
        predicted_action_np,

    absolute_error=
        absolute_error,

    squared_error=
        squared_error,
)


# =========================================================
# 16. Final Summary
# =========================================================

print()
print(
    "===== Dataset v3 Iteration Evaluation Complete ====="
)

print(
    "Overall Report:",
    overall_report_path
)

print(
    "Per-joint Report:",
    joint_report_path
)

print(
    "Per-episode Report:",
    episode_report_path
)

print(
    "Prediction Data:",
    prediction_path
)

print()
print(
    "Dataset v3 Iteration Evaluation: PASS ✅"
)