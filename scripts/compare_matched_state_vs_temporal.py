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
# 2. Paths / Configuration
# =========================================================

TEST_PATH = (
    "dataset_v1/test.npz"
)

MATCHED_MODEL_PATH = (
    "models/state_policy_matched_best.pt"
)

TEMPORAL_MODEL_PATH = (
    "models/temporal_state_policy_best.pt"
)

REPORT_PATH = (
    "reports/"
    "matched_state_vs_temporal_final_comparison.csv"
)

PREDICTION_PATH = (
    "reports/"
    "matched_vs_temporal_predictions.npz"
)

TARGET_DIM = 12


# =========================================================
# 3. Load Test Data
# =========================================================

test = np.load(
    TEST_PATH
)

state = test[
    "state"
].astype(
    np.float32
)

action = test[
    "action"
].astype(
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
# 4. Build Exact Same 1995 Test Frames
# =========================================================

matched_state = []

temporal_feature = []

target_action = []

valid_indices = []


for i in range(
    1,
    len(state)
):

    # Remove episode start
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


    matched_state.append(
        state[i]
    )


    temporal_feature.append(
        np.concatenate(
            [
                state[i],
                delta_state,
            ]
        )
    )


    target_action.append(
        action[i]
    )


    valid_indices.append(
        i
    )


matched_state = np.asarray(
    matched_state,
    dtype=np.float32
)

temporal_feature = np.asarray(
    temporal_feature,
    dtype=np.float32
)

target_action = np.asarray(
    target_action,
    dtype=np.float32
)

valid_indices = np.asarray(
    valid_indices,
    dtype=np.int64
)


assert matched_state.shape == (
    1995,
    14
)

assert temporal_feature.shape == (
    1995,
    28
)

assert target_action.shape == (
    1995,
    14
)


episode_valid = (
    episode_index[
        valid_indices
    ]
)

frame_valid = (
    frame_index[
        valid_indices
    ]
)


print(
    "Same-frame Test Construction: PASS ✅"
)


# =========================================================
# 5. Load Models
# =========================================================

matched_model = StatePolicy()

matched_model.load_state_dict(
    torch.load(
        MATCHED_MODEL_PATH,
        map_location="cpu"
    )
)

matched_model.eval()


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
# 6. Forward Pass
# =========================================================

matched_tensor = torch.tensor(
    matched_state,
    dtype=torch.float32
)

temporal_tensor = torch.tensor(
    temporal_feature,
    dtype=torch.float32
)


with torch.no_grad():

    matched_prediction = (
        matched_model(
            matched_tensor
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


assert matched_prediction.shape == (
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
# 7. Errors
# =========================================================

matched_error = np.abs(
    matched_prediction
    -
    target_action
)

temporal_error = np.abs(
    temporal_prediction
    -
    target_action
)


matched_squared_error = (
    matched_prediction
    -
    target_action
) ** 2

temporal_squared_error = (
    temporal_prediction
    -
    target_action
) ** 2


matched_frame_mae = np.mean(
    matched_error,
    axis=1
)

temporal_frame_mae = np.mean(
    temporal_error,
    axis=1
)


# =========================================================
# 8. Dim12 Dynamics
# =========================================================

dim12_change = np.abs(
    action[
        valid_indices,
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


high_dynamic_mask = (
    dim12_change
    >=
    dim12_threshold
)


normal_dynamic_mask = (
    ~high_dynamic_mask
)


# =========================================================
# 9. Episode / Target Region Masks
# =========================================================

episode49_mask = (
    episode_valid
    ==
    49
)


target_region_mask = (
    (episode_valid == 49)
    &
    (frame_valid >= 292)
    &
    (frame_valid <= 300)
)


frame294_mask = (
    (episode_valid == 49)
    &
    (frame_valid == 294)
)


# =========================================================
# 10. Helper Functions
# =========================================================

def pct_change(
    baseline,
    new
):

    return (
        (new - baseline)
        /
        baseline
        *
        100
    )


def frame_metric(
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


def joint_metric(
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
# 11. Metrics
# =========================================================

metrics = {}


metrics[
    "overall_mae"
] = (

    float(
        np.mean(
            matched_error
        )
    ),

    float(
        np.mean(
            temporal_error
        )
    ),
)


metrics[
    "overall_mse"
] = (

    float(
        np.mean(
            matched_squared_error
        )
    ),

    float(
        np.mean(
            temporal_squared_error
        )
    ),
)


metrics[
    "dim12_overall_mae"
] = (

    joint_metric(
        matched_error,
        TARGET_DIM
    ),

    joint_metric(
        temporal_error,
        TARGET_DIM
    ),
)


metrics[
    "dim12_high_dynamic_mae"
] = (

    joint_metric(
        matched_error,
        TARGET_DIM,
        high_dynamic_mask
    ),

    joint_metric(
        temporal_error,
        TARGET_DIM,
        high_dynamic_mask
    ),
)


metrics[
    "dim12_normal_dynamic_mae"
] = (

    joint_metric(
        matched_error,
        TARGET_DIM,
        normal_dynamic_mask
    ),

    joint_metric(
        temporal_error,
        TARGET_DIM,
        normal_dynamic_mask
    ),
)


metrics[
    "episode49_mae"
] = (

    frame_metric(
        matched_frame_mae,
        episode49_mask
    ),

    frame_metric(
        temporal_frame_mae,
        episode49_mask
    ),
)


metrics[
    "episode49_frames292_300_mae"
] = (

    frame_metric(
        matched_frame_mae,
        target_region_mask
    ),

    frame_metric(
        temporal_frame_mae,
        target_region_mask
    ),
)


metrics[
    "episode49_frames292_300_dim12_mae"
] = (

    joint_metric(
        matched_error,
        TARGET_DIM,
        target_region_mask
    ),

    joint_metric(
        temporal_error,
        TARGET_DIM,
        target_region_mask
    ),
)


metrics[
    "episode49_frame294_mae"
] = (

    frame_metric(
        matched_frame_mae,
        frame294_mask
    ),

    frame_metric(
        temporal_frame_mae,
        frame294_mask
    ),
)


metrics[
    "episode49_frame294_dim12_error"
] = (

    joint_metric(
        matched_error,
        TARGET_DIM,
        frame294_mask
    ),

    joint_metric(
        temporal_error,
        TARGET_DIM,
        frame294_mask
    ),
)


# =========================================================
# 12. Print Final Controlled Comparison
# =========================================================

print()

print("=" * 90)

print(
    "FINAL CONTROLLED EXPERIMENT"
)

print(
    "Matched State-only vs Temporal State"
)

print("=" * 90)


for metric, (
    matched_value,
    temporal_value
) in metrics.items():


    change = pct_change(
        matched_value,
        temporal_value
    )


    print()

    print(
        metric
    )


    print(
        "Matched State-only:",
        matched_value
    )


    print(
        "Temporal State:",
        temporal_value
    )


    print(
        "Change:",
        f"{change:+.2f}%"
    )


# =========================================================
# 13. Save Report
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


    writer = csv.writer(
        f
    )


    writer.writerow(
        [
            "metric",
            "matched_state_only",
            "temporal_state",
            "change_percent",
        ]
    )


    for metric, (
        matched_value,
        temporal_value
    ) in metrics.items():


        writer.writerow(
            [
                metric,
                matched_value,
                temporal_value,
                pct_change(
                    matched_value,
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
            "dim12_high_dynamic_frames",
            int(
                np.sum(
                    high_dynamic_mask
                )
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
# 14. Save Predictions
# =========================================================

np.savez(
    PREDICTION_PATH,

    original_test_index=
        valid_indices,

    episode_index=
        episode_valid,

    frame_index=
        frame_valid,

    ground_truth_action=
        target_action,

    matched_prediction=
        matched_prediction,

    temporal_prediction=
        temporal_prediction,

    matched_absolute_error=
        matched_error,

    temporal_absolute_error=
        temporal_error,
)


# =========================================================
# 15. Final Check
# =========================================================

print()
print("=" * 90)

print(
    "Same Test Frames:",
    len(
        valid_indices
    )
)

print(
    "Dim12 High-dynamic Frames:",
    int(
        np.sum(
            high_dynamic_mask
        )
    )
)

print()

print(
    "Report:",
    REPORT_PATH
)

print()

print(
    "Final Controlled Evaluation: PASS ✅"
)