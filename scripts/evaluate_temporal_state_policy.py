import csv
import os

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# 1. Model Definitions
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


class TemporalStatePolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 14),
        )

    def forward(self, feature):

        return self.net(feature)


# =========================================================
# 2. Paths
# =========================================================

TEST_PATH = "dataset_v1/test.npz"

STATE_MODEL_PATH = (
    "models/state_policy_best.pt"
)

TEMPORAL_MODEL_PATH = (
    "models/temporal_state_policy_best.pt"
)

REPORT_PATH = (
    "reports/"
    "state_vs_temporal_same_frame_comparison.csv"
)

PREDICTION_PATH = (
    "reports/"
    "temporal_state_test_predictions.npz"
)


TARGET_DIM = 12


# =========================================================
# 3. Load Test Dataset
# =========================================================

test = np.load(
    TEST_PATH
)

state = test["state"].astype(
    np.float32
)

action = test["action"].astype(
    np.float32
)

episode_index = test[
    "episode_index"
]

frame_index = test[
    "frame_index"
]


assert state.shape == (
    2000,
    14
)

assert action.shape == (
    2000,
    14
)


print(
    "Test Dataset Loading Check: PASS ✅"
)


# =========================================================
# 4. Build Temporal Test Features
#
# Only keep frames that have t-1
# =========================================================

temporal_features = []

temporal_actions = []

valid_indices = []


for i in range(
    1,
    len(state)
):

    if (
        episode_index[i]
        !=
        episode_index[i - 1]
    ):
        continue


    delta_state = (
        state[i]
        -
        state[i - 1]
    )


    feature = np.concatenate(
        [
            state[i],
            delta_state,
        ]
    )


    temporal_features.append(
        feature
    )

    temporal_actions.append(
        action[i]
    )

    valid_indices.append(
        i
    )


temporal_features = np.asarray(
    temporal_features,
    dtype=np.float32
)

temporal_actions = np.asarray(
    temporal_actions,
    dtype=np.float32
)

valid_indices = np.asarray(
    valid_indices,
    dtype=np.int64
)


assert temporal_features.shape == (
    1995,
    28
)

assert temporal_actions.shape == (
    1995,
    14
)


print(
    "Temporal Test Feature Check: PASS ✅"
)


# =========================================================
# 5. Same 1995 Frames for State-only Baseline
# =========================================================

state_same_frames = (
    state[
        valid_indices
    ]
)

action_same_frames = (
    action[
        valid_indices
    ]
)

episode_same_frames = (
    episode_index[
        valid_indices
    ]
)

frame_same_frames = (
    frame_index[
        valid_indices
    ]
)


assert state_same_frames.shape == (
    1995,
    14
)


# =========================================================
# 6. Load Models
# =========================================================

state_model = StatePolicy()

state_model.load_state_dict(
    torch.load(
        STATE_MODEL_PATH,
        map_location="cpu"
    )
)

state_model.eval()


temporal_model = TemporalStatePolicy()

temporal_model.load_state_dict(
    torch.load(
        TEMPORAL_MODEL_PATH,
        map_location="cpu"
    )
)

temporal_model.eval()


print(
    "Model Loading Check: PASS ✅"
)


# =========================================================
# 7. Forward Pass
# =========================================================

state_tensor = torch.tensor(
    state_same_frames,
    dtype=torch.float32
)

temporal_tensor = torch.tensor(
    temporal_features,
    dtype=torch.float32
)


with torch.no_grad():

    state_prediction = (
        state_model(
            state_tensor
        )
        .cpu()
        .numpy()
    )


    temporal_prediction = (
        temporal_model(
            temporal_tensor
        )
        .cpu()
        .numpy()
    )


assert state_prediction.shape == (
    1995,
    14
)

assert temporal_prediction.shape == (
    1995,
    14
)


print(
    "Forward Pass Check: PASS ✅"
)


# =========================================================
# 8. Error
# =========================================================

state_error = np.abs(
    state_prediction
    -
    action_same_frames
)

temporal_error = np.abs(
    temporal_prediction
    -
    action_same_frames
)


state_squared_error = (
    state_prediction
    -
    action_same_frames
) ** 2

temporal_squared_error = (
    temporal_prediction
    -
    action_same_frames
) ** 2


state_frame_mae = np.mean(
    state_error,
    axis=1
)

temporal_frame_mae = np.mean(
    temporal_error,
    axis=1
)


# =========================================================
# 9. Dim12 Dynamics
# =========================================================

dim12_change = np.abs(
    action_same_frames[
        :,
        TARGET_DIM
    ]
    -
    action[
        valid_indices - 1,
        TARGET_DIM
    ]
)


dim12_threshold = np.percentile(
    dim12_change,
    90
)


dim12_high_mask = (
    dim12_change
    >=
    dim12_threshold
)


dim12_normal_mask = (
    ~dim12_high_mask
)


# =========================================================
# 10. Target Failure Masks
# =========================================================

episode49_mask = (
    episode_same_frames
    ==
    49
)


target_region_mask = (
    (episode_same_frames == 49)
    &
    (frame_same_frames >= 292)
    &
    (frame_same_frames <= 300)
)


frame294_mask = (
    (episode_same_frames == 49)
    &
    (frame_same_frames == 294)
)


# =========================================================
# 11. Helper Functions
# =========================================================

def pct_change(
    baseline,
    new_value
):

    return (
        (new_value - baseline)
        /
        baseline
        *
        100
    )


def mean_frame_mae(
    values,
    mask=None
):

    if mask is None:

        return float(
            np.mean(
                values
            )
        )

    return float(
        np.mean(
            values[
                mask
            ]
        )
    )


def mean_joint_mae(
    error,
    dim,
    mask=None
):

    if mask is None:

        return float(
            np.mean(
                error[
                    :,
                    dim
                ]
            )
        )

    return float(
        np.mean(
            error[
                mask,
                dim
            ]
        )
    )


# =========================================================
# 12. Metrics
# =========================================================

metrics = {}


metrics[
    "overall_mae"
] = (
    float(
        np.mean(
            state_error
        )
    ),
    float(
        np.mean(
            temporal_error
        )
    )
)


metrics[
    "overall_mse"
] = (
    float(
        np.mean(
            state_squared_error
        )
    ),
    float(
        np.mean(
            temporal_squared_error
        )
    )
)


metrics[
    "dim12_overall_mae"
] = (
    mean_joint_mae(
        state_error,
        TARGET_DIM
    ),
    mean_joint_mae(
        temporal_error,
        TARGET_DIM
    )
)


metrics[
    "dim12_high_dynamic_mae"
] = (
    mean_joint_mae(
        state_error,
        TARGET_DIM,
        dim12_high_mask
    ),
    mean_joint_mae(
        temporal_error,
        TARGET_DIM,
        dim12_high_mask
    )
)


metrics[
    "dim12_normal_dynamic_mae"
] = (
    mean_joint_mae(
        state_error,
        TARGET_DIM,
        dim12_normal_mask
    ),
    mean_joint_mae(
        temporal_error,
        TARGET_DIM,
        dim12_normal_mask
    )
)


metrics[
    "episode49_mae"
] = (
    mean_frame_mae(
        state_frame_mae,
        episode49_mask
    ),
    mean_frame_mae(
        temporal_frame_mae,
        episode49_mask
    )
)


metrics[
    "episode49_frames292_300_mae"
] = (
    mean_frame_mae(
        state_frame_mae,
        target_region_mask
    ),
    mean_frame_mae(
        temporal_frame_mae,
        target_region_mask
    )
)


metrics[
    "episode49_frames292_300_dim12_mae"
] = (
    mean_joint_mae(
        state_error,
        TARGET_DIM,
        target_region_mask
    ),
    mean_joint_mae(
        temporal_error,
        TARGET_DIM,
        target_region_mask
    )
)


metrics[
    "episode49_frame294_mae"
] = (
    mean_frame_mae(
        state_frame_mae,
        frame294_mask
    ),
    mean_frame_mae(
        temporal_frame_mae,
        frame294_mask
    )
)


metrics[
    "episode49_frame294_dim12_error"
] = (
    mean_joint_mae(
        state_error,
        TARGET_DIM,
        frame294_mask
    ),
    mean_joint_mae(
        temporal_error,
        TARGET_DIM,
        frame294_mask
    )
)


# =========================================================
# 13. Print Results
# =========================================================

print()
print("=" * 80)
print(
    "State-only vs Temporal State"
)
print(
    "Same-frame Neural Network Evaluation"
)
print("=" * 80)


for metric, (
    state_value,
    temporal_value
) in metrics.items():

    change = pct_change(
        state_value,
        temporal_value
    )


    print()

    print(
        metric
    )

    print(
        "State-only:",
        state_value
    )

    print(
        "Temporal:",
        temporal_value
    )

    print(
        "Change:",
        f"{change:+.2f}%"
    )


# =========================================================
# 14. Save Report
# =========================================================

os.makedirs(
    "reports",
    exist_ok=True
)


with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "metric",
            "state_only",
            "temporal_state",
            "change_percent",
        ]
    )


    for metric, (
        state_value,
        temporal_value
    ) in metrics.items():

        writer.writerow(
            [
                metric,
                state_value,
                temporal_value,
                pct_change(
                    state_value,
                    temporal_value
                ),
            ]
        )


    writer.writerow(
        [
            "same_test_frames",
            len(
                valid_indices
            ),
            "",
            "",
        ]
    )


    writer.writerow(
        [
            "dim12_high_dynamic_threshold",
            dim12_threshold,
            "",
            "",
        ]
    )


# =========================================================
# 15. Save Temporal Predictions
# =========================================================

np.savez(
    PREDICTION_PATH,

    original_test_index=
        valid_indices,

    episode_index=
        episode_same_frames,

    frame_index=
        frame_same_frames,

    ground_truth_action=
        action_same_frames,

    predicted_action=
        temporal_prediction,

    absolute_error=
        temporal_error,

    squared_error=
        temporal_squared_error,
)


# =========================================================
# 16. Final Summary
# =========================================================

print()
print("=" * 80)

print(
    "Same Test Frames:",
    len(
        valid_indices
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
    REPORT_PATH
)

print(
    "Predictions:",
    PREDICTION_PATH
)

print()

print(
    "State vs Temporal Evaluation: PASS ✅"
)