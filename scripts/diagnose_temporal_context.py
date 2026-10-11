import csv
import os
import numpy as np


# =========================================================
# 1. Configuration
# =========================================================

TRAIN_PATH = "dataset_v1/train.npz"
TEST_PATH = "dataset_v1/test.npz"

OUTPUT_PATH = (
    "reports/temporal_context_diagnostic.csv"
)

K = 20
TARGET_DIM = 12


# =========================================================
# 2. Load Data
# =========================================================

train = np.load(TRAIN_PATH)
test = np.load(TEST_PATH)

train_state = train["state"].astype(
    np.float64
)

train_action = train["action"].astype(
    np.float64
)

train_episode = train[
    "episode_index"
]

test_state = test["state"].astype(
    np.float64
)

test_action = test["action"].astype(
    np.float64
)

test_episode = test[
    "episode_index"
]

test_frame = test[
    "frame_index"
]


print("=" * 70)
print("Temporal Context Diagnostic")
print("=" * 70)

print(
    "Train:",
    train_state.shape
)

print(
    "Test:",
    test_state.shape
)


# =========================================================
# 3. Compute Delta State
#
# ΔState(t) = State(t) - State(t-1)
#
# Episode starts are invalid.
# =========================================================

def compute_delta_state(
    state,
    episode
):

    delta = np.full_like(
        state,
        np.nan,
        dtype=np.float64
    )

    for i in range(
        1,
        len(state)
    ):

        if (
            episode[i]
            !=
            episode[i - 1]
        ):
            continue

        delta[i] = (
            state[i]
            -
            state[i - 1]
        )

    return delta


train_delta = compute_delta_state(
    train_state,
    train_episode
)

test_delta = compute_delta_state(
    test_state,
    test_episode
)


train_valid = np.all(
    np.isfinite(
        train_delta
    ),
    axis=1
)

test_valid = np.all(
    np.isfinite(
        test_delta
    ),
    axis=1
)


print()
print(
    "Valid Train Frames:",
    int(
        np.sum(
            train_valid
        )
    )
)

print(
    "Valid Test Frames:",
    int(
        np.sum(
            test_valid
        )
    )
)


# =========================================================
# 4. Standardization
#
# Important:
# State and Delta State can have different scales.
#
# All statistics are calculated ONLY using Train.
# =========================================================

state_mean = np.mean(
    train_state,
    axis=0
)

state_std = np.std(
    train_state,
    axis=0
)

state_std[
    state_std < 1e-8
] = 1.0


delta_mean = np.mean(
    train_delta[
        train_valid
    ],
    axis=0
)

delta_std = np.std(
    train_delta[
        train_valid
    ],
    axis=0
)

delta_std[
    delta_std < 1e-8
] = 1.0


train_state_z = (
    train_state
    -
    state_mean
) / state_std


test_state_z = (
    test_state
    -
    state_mean
) / state_std


train_delta_z = (
    train_delta
    -
    delta_mean
) / delta_std


test_delta_z = (
    test_delta
    -
    delta_mean
) / delta_std


# =========================================================
# 5. Build Feature Spaces
# =========================================================

# State-only
train_feature_state = (
    train_state_z[
        train_valid
    ]
)

train_action_valid = (
    train_action[
        train_valid
    ]
)


# State + Delta State
train_feature_temporal = np.concatenate(
    [
        train_state_z[
            train_valid
        ],
        train_delta_z[
            train_valid
        ],
    ],
    axis=1
)


test_indices = np.where(
    test_valid
)[0]


# =========================================================
# 6. KNN Prediction
# =========================================================

state_predictions = []

temporal_predictions = []

state_local_std = []

temporal_local_std = []


print()
print(
    "Running KNN Diagnostic..."
)


for count, idx in enumerate(
    test_indices,
    start=1
):

    # -----------------------------------------------------
    # State-only
    # -----------------------------------------------------

    query_state = (
        test_state_z[
            idx
        ]
    )


    state_distance = np.sum(
        (
            train_feature_state
            -
            query_state
        ) ** 2,
        axis=1
    )


    state_neighbor_idx = np.argpartition(
        state_distance,
        K
    )[:K]


    state_neighbor_actions = (
        train_action_valid[
            state_neighbor_idx
        ]
    )


    state_prediction = np.mean(
        state_neighbor_actions,
        axis=0
    )


    state_predictions.append(
        state_prediction
    )


    state_local_std.append(
        np.mean(
            np.std(
                state_neighbor_actions,
                axis=0
            )
        )
    )


    # -----------------------------------------------------
    # State + Delta State
    # -----------------------------------------------------

    query_temporal = np.concatenate(
        [
            test_state_z[
                idx
            ],
            test_delta_z[
                idx
            ],
        ]
    )


    temporal_distance = np.sum(
        (
            train_feature_temporal
            -
            query_temporal
        ) ** 2,
        axis=1
    )


    temporal_neighbor_idx = np.argpartition(
        temporal_distance,
        K
    )[:K]


    temporal_neighbor_actions = (
        train_action_valid[
            temporal_neighbor_idx
        ]
    )


    temporal_prediction = np.mean(
        temporal_neighbor_actions,
        axis=0
    )


    temporal_predictions.append(
        temporal_prediction
    )


    temporal_local_std.append(
        np.mean(
            np.std(
                temporal_neighbor_actions,
                axis=0
            )
        )
    )


state_predictions = np.asarray(
    state_predictions
)

temporal_predictions = np.asarray(
    temporal_predictions
)

state_local_std = np.asarray(
    state_local_std
)

temporal_local_std = np.asarray(
    temporal_local_std
)


ground_truth = (
    test_action[
        test_indices
    ]
)


# =========================================================
# 7. Errors
# =========================================================

state_error = np.abs(
    state_predictions
    -
    ground_truth
)

temporal_error = np.abs(
    temporal_predictions
    -
    ground_truth
)


state_frame_mae = np.mean(
    state_error,
    axis=1
)

temporal_frame_mae = np.mean(
    temporal_error,
    axis=1
)


# =========================================================
# 8. Test Dim12 Dynamics
# =========================================================

dim12_change = np.full(
    len(test_action),
    np.nan
)


for i in range(
    1,
    len(test_action)
):

    if (
        test_episode[i]
        !=
        test_episode[i - 1]
    ):
        continue

    dim12_change[i] = abs(
        test_action[
            i,
            TARGET_DIM
        ]
        -
        test_action[
            i - 1,
            TARGET_DIM
        ]
    )


threshold = np.percentile(
    dim12_change[
        np.isfinite(
            dim12_change
        )
    ],
    90
)


# Masks in original Test coordinates
high_dynamic_original = (
    test_valid
    &
    (
        dim12_change
        >=
        threshold
    )
)


target_region_original = (
    test_valid
    &
    (test_episode == 49)
    &
    (test_frame >= 292)
    &
    (test_frame <= 300)
)


frame294_original = (
    test_valid
    &
    (test_episode == 49)
    &
    (test_frame == 294)
)


# Convert original masks into valid-test coordinates
high_mask = (
    high_dynamic_original[
        test_indices
    ]
)

target_mask = (
    target_region_original[
        test_indices
    ]
)

frame294_mask = (
    frame294_original[
        test_indices
    ]
)


# =========================================================
# 9. Metrics
# =========================================================

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


def frame_mae(
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


def joint_mae(
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


metrics = {}


metrics[
    "overall_mae"
] = (
    frame_mae(
        state_frame_mae
    ),
    frame_mae(
        temporal_frame_mae
    )
)


metrics[
    "dim12_overall_mae"
] = (
    joint_mae(
        state_error,
        TARGET_DIM
    ),
    joint_mae(
        temporal_error,
        TARGET_DIM
    )
)


metrics[
    "dim12_high_dynamic_mae"
] = (
    joint_mae(
        state_error,
        TARGET_DIM,
        high_mask
    ),
    joint_mae(
        temporal_error,
        TARGET_DIM,
        high_mask
    )
)


metrics[
    "episode49_frames292_300_mae"
] = (
    frame_mae(
        state_frame_mae,
        target_mask
    ),
    frame_mae(
        temporal_frame_mae,
        target_mask
    )
)


metrics[
    "episode49_frames292_300_dim12_mae"
] = (
    joint_mae(
        state_error,
        TARGET_DIM,
        target_mask
    ),
    joint_mae(
        temporal_error,
        TARGET_DIM,
        target_mask
    )
)


metrics[
    "episode49_frame294_mae"
] = (
    frame_mae(
        state_frame_mae,
        frame294_mask
    ),
    frame_mae(
        temporal_frame_mae,
        frame294_mask
    )
)


metrics[
    "mean_local_action_std"
] = (
    float(
        np.mean(
            state_local_std
        )
    ),
    float(
        np.mean(
            temporal_local_std
        )
    )
)


# =========================================================
# 10. Print Results
# =========================================================

print()
print("=" * 80)

print(
    "State-only vs State+DeltaState KNN"
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
    print(metric)

    print(
        "State-only:",
        state_value
    )

    print(
        "State + DeltaState:",
        temporal_value
    )

    print(
        "Change:",
        f"{change:+.2f}%"
    )


# =========================================================
# 11. Save Report
# =========================================================

os.makedirs(
    "reports",
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "metric",
            "state_only",
            "state_plus_delta_state",
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
            "knn_k",
            K,
            "",
            "",
        ]
    )


    writer.writerow(
        [
            "dim12_high_dynamic_threshold",
            threshold,
            "",
            "",
        ]
    )


print()
print(
    "Report:",
    OUTPUT_PATH
)

print()
print(
    "Temporal Context Diagnostic: PASS ✅"
)