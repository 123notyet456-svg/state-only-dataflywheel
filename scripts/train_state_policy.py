from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import csv
import os


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


if __name__ == "__main__":

    # =========================================================
    # 2. Dataset Loading
    # =========================================================

    train_data = np.load("dataset_v1/train.npz")
    val_data = np.load("dataset_v1/val.npz")

    train_state = torch.tensor(
        train_data["state"],
        dtype=torch.float32
    )

    train_action = torch.tensor(
        train_data["action"],
        dtype=torch.float32
    )

    val_state = torch.tensor(
        val_data["state"],
        dtype=torch.float32
    )

    val_action = torch.tensor(
        val_data["action"],
        dtype=torch.float32
    )

    print("Train state shape:", train_state.shape)
    print("Train action shape:", train_action.shape)
    print("Val state shape:", val_state.shape)
    print("Val action shape:", val_action.shape)

    assert train_state.shape == (16000, 14)
    assert train_action.shape == (16000, 14)
    assert val_state.shape == (2000, 14)
    assert val_action.shape == (2000, 14)

    print("Dataset Loading Check: PASS ✅")


    # =========================================================
    # 3. Training Configuration
    # =========================================================

    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50

    model = StatePolicy()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # =========================================================
    # 4. DataLoader
    # =========================================================

    train_dataset = TensorDataset(
        train_state,
        train_action
    )

    val_dataset = TensorDataset(
        val_state,
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
    print("===== Training Configuration =====")
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Learning Rate:", learning_rate)
    print("Batch Size:", batch_size)
    print("Epochs:", num_epochs)
    print("Training Setup Check: PASS ✅")


    # =========================================================
    # 5. Output Directory
    # =========================================================

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    model_path = "models/state_policy_best.pt"
    history_path = "reports/baseline_training_history.csv"


    # =========================================================
    # 6. Training Loop
    # =========================================================

    training_history = []

    best_val_loss = float("inf")
    best_epoch = -1

    print()
    print("===== Baseline Training Start =====")

    for epoch in range(1, num_epochs + 1):

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        model.train()

        total_train_loss = 0.0

        for batch_state, batch_action in train_loader:

            predicted_action = model(batch_state)

            loss = criterion(
                predicted_action,
                batch_action
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += (
                loss.item() * batch_state.size(0)
            )

        average_train_loss = (
            total_train_loss / len(train_dataset)
        )


        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        model.eval()

        total_val_loss = 0.0

        with torch.no_grad():

            for val_batch_state, val_batch_action in val_loader:

                val_predicted_action = model(
                    val_batch_state
                )

                val_loss = criterion(
                    val_predicted_action,
                    val_batch_action
                )

                total_val_loss += (
                    val_loss.item()
                    * val_batch_state.size(0)
                )

        average_val_loss = (
            total_val_loss / len(val_dataset)
        )


        # -----------------------------------------------------
        # Save History
        # -----------------------------------------------------

        training_history.append({
            "epoch": epoch,
            "train_loss": average_train_loss,
            "val_loss": average_val_loss,
        })


        # -----------------------------------------------------
        # Save Best Model
        # -----------------------------------------------------

        if average_val_loss < best_val_loss:

            best_val_loss = average_val_loss
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                model_path
            )


        # -----------------------------------------------------
        # Print Epoch Result
        # -----------------------------------------------------

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {average_train_loss:.8f} | "
            f"Val Loss: {average_val_loss:.8f}"
        )


    # =========================================================
    # 7. Save Training History CSV
    # =========================================================

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
                "val_loss"
            ]
        )

        writer.writeheader()
        writer.writerows(training_history)


    # =========================================================
    # 8. Final Summary
    # =========================================================

    print()
    print("===== Baseline Training Complete =====")
    print("Best Epoch:", best_epoch)
    print("Best Validation Loss:", best_val_loss)
    print("Best Model:", model_path)
    print("Training History:", history_path)

    assert os.path.exists(model_path)
    assert os.path.exists(history_path)
    assert np.isfinite(best_val_loss)

    print("Baseline Training Check: PASS ✅")