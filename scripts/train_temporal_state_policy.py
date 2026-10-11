from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch
import torch.nn as nn

import csv
import os


# =========================================================
# 1. Model Definition
#
# Input:
# State(t)        = 14 dims
# Delta State(t)  = 14 dims
#
# Total Input:
# 28 dims
#
# Output:
# Action(t)       = 14 dims
# =========================================================

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


    def forward(
        self,
        state
    ):

        return self.net(
            state
        )


# =========================================================
# 2. Delta State Builder
# =========================================================

def build_temporal_features(
    state,
    action,
    episode_index
):

    features = []
    targets = []

    kept_indices = []


    for i in range(
        1,
        len(state)
    ):

        # Do not cross episode boundary
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


        temporal_feature = np.concatenate(
            [
                state[i],
                delta_state,
            ]
        )


        features.append(
            temporal_feature
        )

        targets.append(
            action[i]
        )

        kept_indices.append(
            i
        )


    return (
        np.asarray(
            features,
            dtype=np.float32
        ),

        np.asarray(
            targets,
            dtype=np.float32
        ),

        np.asarray(
            kept_indices,
            dtype=np.int64
        ),
    )


# =========================================================
# 3. Main
# =========================================================

if __name__ == "__main__":


    # =====================================================
    # 4. Dataset Loading
    # =====================================================

    train_data = np.load(
        "dataset_v1/train.npz"
    )

    val_data = np.load(
        "dataset_v1/val.npz"
    )


    train_feature_np, train_action_np, train_indices = (
        build_temporal_features(

            train_data["state"],

            train_data["action"],

            train_data[
                "episode_index"
            ],
        )
    )


    val_feature_np, val_action_np, val_indices = (
        build_temporal_features(

            val_data["state"],

            val_data["action"],

            val_data[
                "episode_index"
            ],
        )
    )


    print(
        "Train temporal feature shape:",
        train_feature_np.shape
    )

    print(
        "Train action shape:",
        train_action_np.shape
    )

    print(
        "Val temporal feature shape:",
        val_feature_np.shape
    )

    print(
        "Val action shape:",
        val_action_np.shape
    )


    assert train_feature_np.shape == (
        15960,
        28
    )

    assert train_action_np.shape == (
        15960,
        14
    )

    assert val_feature_np.shape == (
        1995,
        28
    )

    assert val_action_np.shape == (
        1995,
        14
    )


    print(
        "Temporal Dataset Construction Check: PASS ✅"
    )


    # =====================================================
    # 5. Convert to Tensor
    # =====================================================

    train_feature = torch.tensor(
        train_feature_np,
        dtype=torch.float32
    )


    train_action = torch.tensor(
        train_action_np,
        dtype=torch.float32
    )


    val_feature = torch.tensor(
        val_feature_np,
        dtype=torch.float32
    )


    val_action = torch.tensor(
        val_action_np,
        dtype=torch.float32
    )


    # =====================================================
    # 6. Training Configuration
    # =====================================================

    learning_rate = 0.001

    batch_size = 64

    num_epochs = 50


    model = TemporalStatePolicy()


    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=learning_rate
    )


    # =====================================================
    # 7. Dataset / DataLoader
    # =====================================================

    train_dataset = TensorDataset(

        train_feature,

        train_action
    )


    val_dataset = TensorDataset(

        val_feature,

        val_action
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False
    )


    print()

    print(
        "===== Temporal Training Configuration ====="
    )

    print(
        "Input Dimensions: 28"
    )

    print(
        "Output Dimensions: 14"
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Val samples:",
        len(val_dataset)
    )

    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Val batches:",
        len(val_loader)
    )

    print(
        "Learning Rate:",
        learning_rate
    )

    print(
        "Batch Size:",
        batch_size
    )

    print(
        "Epochs:",
        num_epochs
    )

    print(
        "Training Setup Check: PASS ✅"
    )


    # =====================================================
    # 8. Output Paths
    # =====================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )


    model_path = (
        "models/"
        "temporal_state_policy_best.pt"
    )


    history_path = (
        "reports/"
        "temporal_state_training_history.csv"
    )


    # =====================================================
    # 9. Training Loop
    # =====================================================

    training_history = []

    best_val_loss = float(
        "inf"
    )

    best_epoch = -1


    print()

    print(
        "===== Temporal State Training Start ====="
    )


    for epoch in range(
        1,
        num_epochs + 1
    ):


        # =================================================
        # Train
        # =================================================

        model.train()

        total_train_loss = 0.0


        for (
            batch_feature,
            batch_action
        ) in train_loader:


            predicted_action = model(
                batch_feature
            )


            loss = criterion(

                predicted_action,

                batch_action
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_train_loss += (

                loss.item()

                *

                batch_feature.size(0)
            )


        average_train_loss = (

            total_train_loss

            /

            len(train_dataset)
        )


        # =================================================
        # Validation
        # =================================================

        model.eval()

        total_val_loss = 0.0


        with torch.no_grad():


            for (
                val_batch_feature,
                val_batch_action
            ) in val_loader:


                val_predicted_action = model(

                    val_batch_feature
                )


                val_loss = criterion(

                    val_predicted_action,

                    val_batch_action
                )


                total_val_loss += (

                    val_loss.item()

                    *

                    val_batch_feature.size(0)
                )


        average_val_loss = (

            total_val_loss

            /

            len(val_dataset)
        )


        # =================================================
        # History
        # =================================================

        training_history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    average_train_loss,

                "val_loss":
                    average_val_loss,
            }
        )


        # =================================================
        # Best Checkpoint
        # =================================================

        if (
            average_val_loss
            <
            best_val_loss
        ):


            best_val_loss = (
                average_val_loss
            )


            best_epoch = (
                epoch
            )


            torch.save(

                model.state_dict(),

                model_path
            )


        # =================================================
        # Epoch Print
        # =================================================

        print(

            f"Epoch "
            f"{epoch:02d}/"
            f"{num_epochs} | "

            f"Train Loss: "
            f"{average_train_loss:.8f} | "

            f"Val Loss: "
            f"{average_val_loss:.8f}"
        )


    # =====================================================
    # 10. Save Training History
    # =====================================================

    with open(

        history_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
            ]
        )


        writer.writeheader()

        writer.writerows(
            training_history
        )


    # =====================================================
    # 11. Final Summary
    # =====================================================

    print()

    print(
        "===== Temporal State Training Complete ====="
    )


    print(
        "Best Epoch:",
        best_epoch
    )


    print(
        "Best Validation Loss:",
        best_val_loss
    )


    print(
        "Best Model:",
        model_path
    )


    print(
        "Training History:",
        history_path
    )


    assert os.path.exists(
        model_path
    )


    assert os.path.exists(
        history_path
    )


    assert np.isfinite(
        best_val_loss
    )


    print(
        "Temporal State Training Check: PASS ✅"
    )