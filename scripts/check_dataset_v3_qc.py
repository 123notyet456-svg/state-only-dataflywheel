from pathlib import Path
import csv
import numpy as np


# =========================================================
# 1. Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

V1_DIR = PROJECT_ROOT / "dataset_v1"
V3_DIR = PROJECT_ROOT / "dataset_v3"
REPORT_DIR = PROJECT_ROOT / "reports"

V1_TRAIN = V1_DIR / "train.npz"
V1_VAL = V1_DIR / "val.npz"
V1_TEST = V1_DIR / "test.npz"

V3_TRAIN = V3_DIR / "train.npz"
V3_VAL = V3_DIR / "val.npz"
V3_TEST = V3_DIR / "test.npz"

SELECTED_REPORT = (
    REPORT_DIR
    / "dataset_v3_dim12_high_dynamics.csv"
)

SUMMARY_REPORT = (
    REPORT_DIR
    / "dataset_v3_qc_summary.csv"
)


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


print("=" * 70)
print("Dataset v3 QC")
print("=" * 70)


# =========================================================
# 2. Load Data
# =========================================================

v1_train = np.load(
    V1_TRAIN,
    allow_pickle=True
)

v3_train = np.load(
    V3_TRAIN,
    allow_pickle=True
)

v1_val = np.load(
    V1_VAL,
    allow_pickle=True
)

v3_val = np.load(
    V3_VAL,
    allow_pickle=True
)

v1_test = np.load(
    V1_TEST,
    allow_pickle=True
)

v3_test = np.load(
    V3_TEST,
    allow_pickle=True
)


# =========================================================
# 3. Shape Check
# =========================================================

expected_frames = 19192

shape_check = (
    v3_train["state"].shape
    == (expected_frames, 14)
    and
    v3_train["action"].shape
    == (expected_frames, 14)
    and
    len(v3_train["episode_index"])
    == expected_frames
    and
    len(v3_train["frame_index"])
    == expected_frames
    and
    len(v3_train["timestamp"])
    == expected_frames
)

print()
print("Shape Check:")
print(pass_fail(shape_check))


# =========================================================
# 4. NaN / Inf Check
# =========================================================

nan_inf_ok = True

for key in [
    "state",
    "action",
    "episode_index",
    "frame_index",
    "timestamp",
]:

    arr = np.asarray(
        v3_train[key]
    )

    if not np.all(
        np.isfinite(arr)
    ):

        nan_inf_ok = False

        print(
            "Invalid values:",
            key
        )


print()
print("NaN / Inf Check:")
print(pass_fail(nan_inf_ok))


# =========================================================
# 5. Original Dataset v1 Preservation
#
# Dataset v3 的前 16000 Frames
# 应该与 Dataset v1 完全一致
# =========================================================

original_frames = len(
    v1_train["state"]
)

preservation_ok = True


for key in v1_train.files:

    arr1 = v1_train[key]
    arr3 = v3_train[key]

    if (
        hasattr(arr1, "shape")
        and
        arr1.ndim > 0
        and
        arr1.shape[0] == original_frames
    ):

        if not np.array_equal(
            arr1,
            arr3[:original_frames]
        ):

            preservation_ok = False

            print(
                "Preservation failed:",
                key
            )


print()
print("Original Dataset v1 Preservation:")
print(
    pass_fail(
        preservation_ok
    )
)


# =========================================================
# 6. Read Selected Dim12 High-dynamics Frames
# =========================================================

selected_pairs = set()


with open(
    SELECTED_REPORT,
    "r",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        pair = (
            int(
                row["episode_index"]
            ),
            int(
                row["frame_index"]
            ),
        )

        selected_pairs.add(
            pair
        )


print()
print("Selected Dim12 High-dynamics Frames:")
print(len(selected_pairs))


selected_count_ok = (
    len(selected_pairs)
    == 1596
)


print()
print("Selected Frame Count Check:")
print(
    pass_fail(
        selected_count_ok
    )
)


# =========================================================
# 7. Count Multiplicity in Dataset v3
# =========================================================

v3_episode = np.asarray(
    v3_train["episode_index"]
)

v3_frame = np.asarray(
    v3_train["frame_index"]
)


frame_counts = {}


for ep, frame in zip(
    v3_episode,
    v3_frame
):

    pair = (
        int(ep),
        int(frame),
    )

    frame_counts[pair] = (
        frame_counts.get(
            pair,
            0
        )
        + 1
    )


selected_correct = 0
selected_wrong = 0

normal_correct = 0
normal_wrong = 0


for pair, count in frame_counts.items():

    if pair in selected_pairs:

        # Original 1 + extra 2
        if count == 3:
            selected_correct += 1

        else:
            selected_wrong += 1

    else:

        if count == 1:
            normal_correct += 1

        else:
            normal_wrong += 1


oversampling_ok = (
    selected_correct
    == 1596
    and
    selected_wrong == 0
    and
    normal_wrong == 0
)


print()
print("Oversampling Check:")
print(
    pass_fail(
        oversampling_ok
    )
)

print()
print(
    "Selected correct:",
    selected_correct
)

print(
    "Selected wrong:",
    selected_wrong
)

print(
    "Normal correct:",
    normal_correct
)

print(
    "Normal wrong:",
    normal_wrong
)


# =========================================================
# 8. Episode Boundary Check
#
# 每个 Episode 第一帧不能因为 Dynamics
# 被错误 Oversample
# =========================================================

episode_boundary_ok = True


for ep in np.unique(
    v1_train["episode_index"]
):

    mask = (
        v1_train["episode_index"]
        == ep
    )

    first_frame = int(
        np.min(
            v1_train[
                "frame_index"
            ][mask]
        )
    )

    pair = (
        int(ep),
        first_frame,
    )

    if (
        frame_counts.get(
            pair,
            0
        )
        != 1
    ):

        episode_boundary_ok = False

        print(
            "Episode start duplicated:",
            pair
        )


print()
print("Episode Boundary Check:")
print(
    pass_fail(
        episode_boundary_ok
    )
)


# =========================================================
# 9. Validation Unchanged
# =========================================================

val_same = True


for key in v1_val.files:

    if not np.array_equal(
        v1_val[key],
        v3_val[key]
    ):

        val_same = False

        print(
            "Validation differs:",
            key
        )


print()
print("Validation Unchanged:")
print(
    pass_fail(
        val_same
    )
)


# =========================================================
# 10. Test Unchanged
# =========================================================

test_same = True


for key in v1_test.files:

    if not np.array_equal(
        v1_test[key],
        v3_test[key]
    ):

        test_same = False

        print(
            "Test differs:",
            key
        )


print()
print("Test Unchanged:")
print(
    pass_fail(
        test_same
    )
)


# =========================================================
# 11. Sampling Ratio
# =========================================================

v1_target_ratio = (
    1596
    / 16000
    * 100
)

v3_target_samples = (
    1596 * 3
)

v3_target_ratio = (
    v3_target_samples
    / 19192
    * 100
)


print()
print(
    "Dim12 High-dynamics Sampling Ratio:"
)

print(
    "Dataset v1:",
    f"{v1_target_ratio:.2f}%"
)

print(
    "Dataset v3:",
    f"{v3_target_ratio:.2f}%"
)


# =========================================================
# 12. Overall QC
# =========================================================

overall_pass = all(
    [
        shape_check,
        nan_inf_ok,
        preservation_ok,
        selected_count_ok,
        oversampling_ok,
        episode_boundary_ok,
        val_same,
        test_same,
    ]
)


print()
print("=" * 70)

print(
    "Overall Dataset v3 QC:",
    pass_fail(
        overall_pass
    )
)

print("=" * 70)


# =========================================================
# 13. Save Report
# =========================================================

rows = [

    [
        "check",
        "result"
    ],

    [
        "shape_check",
        pass_fail(
            shape_check
        )
    ],

    [
        "nan_inf_check",
        pass_fail(
            nan_inf_ok
        )
    ],

    [
        "original_train_preservation",
        pass_fail(
            preservation_ok
        )
    ],

    [
        "selected_frame_count",
        pass_fail(
            selected_count_ok
        )
    ],

    [
        "oversampling_check",
        pass_fail(
            oversampling_ok
        )
    ],

    [
        "episode_boundary_check",
        pass_fail(
            episode_boundary_ok
        )
    ],

    [
        "validation_unchanged",
        pass_fail(
            val_same
        )
    ],

    [
        "test_unchanged",
        pass_fail(
            test_same
        )
    ],

    [
        "selected_high_dynamic_frames",
        len(
            selected_pairs
        )
    ],

    [
        "selected_correct_count",
        selected_correct
    ],

    [
        "selected_wrong_count",
        selected_wrong
    ],

    [
        "normal_wrong_count",
        normal_wrong
    ],

    [
        "dataset_v1_target_ratio_percent",
        v1_target_ratio
    ],

    [
        "dataset_v3_target_sampling_ratio_percent",
        v3_target_ratio
    ],

    [
        "overall_qc",
        pass_fail(
            overall_pass
        )
    ],
]


with open(
    SUMMARY_REPORT,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerows(
        rows
    )


print()
print("QC Report:")
print(
    SUMMARY_REPORT
)