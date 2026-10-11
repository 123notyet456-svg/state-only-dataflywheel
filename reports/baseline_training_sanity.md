# State-only Baseline Training Sanity Check

## Model

StatePolicy MLP

Architecture:

14 → 128 → 128 → 14

## Dataset

Training samples:

16000 Frames

Batch Size:

64

Number of Batches:

250

## One Batch Training Check

Predicted Action Shape:

(64, 14)

MSE Loss:

0.2352696955

Result:

PASS

## One Batch Overfit Sanity Check

A temporary StatePolicy model was trained repeatedly on the same batch for 50 steps.

Initial Loss:

0.2420955002

Final Loss:

0.0079553770

Result:

PASS

## Purpose

Verify that the complete Behavior Cloning training pipeline can learn before starting full-dataset training.

The check confirms:

- Dataset loading works
- DataLoader works
- Forward pass works
- MSE loss calculation works
- Backward propagation works
- Adam optimizer updates model parameters
- Model loss can decrease substantially on a fixed batch

## Conclusion

State-only Behavior Cloning training pipeline is ready for full training.