from pathlib import Path
import csv
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

V1_DIR = PROJECT_ROOT / "dataset_v1"
V2_DIR = PROJECT_ROOT / "dataset_v2"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

V1_TRAIN = V1_DIR / "train.npz"
V1_VAL = V1_DIR / "val.npz"
V1_TEST = V1_DIR / "test.npz"

V2_TRAIN = V2_DIR / "train.npz"
V2_VAL = V2_DIR / "val.npz"
V2_TEST = V2_DIR / "test.npz"

HIGH_DYNAMIC_REPORT = (
    REPORT_DIR / "dataset_v2_high_dynamics.csv"
)

SUMMARY_REPORT = (
    REPORT_DIR / "dataset_v2_qc_summary.csv"
)


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


print("=" * 70)
print("Dataset v2 QC")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------

v1_train = np.load(V1_TRAIN, allow_pickle=True)
v2_train = np.load(V2_TRAIN, allow_pickle=True)

v1_val = np.load(V1_VAL, allow_pickle=True)
v2_val = np.load(V2_VAL, allow_pickle=True)

v1_test = np.load(V1_TEST, allow_pickle=True)
v2_test = np.load(V2_TEST, allow_pickle=True)


# ------------------------------------------------------------
# 2. Basic Shape Check
# ------------------------------------------------------------

expected_v2_train_frames = 19192

shape_check = (
    v2_train["state"].shape == (expected_v2_train_frames, 14)
    and
    v2_train["action"].shape == (expected_v2_train_frames, 14)
    and
    len(v2_train["episode_index"]) == expected_v2_train_frames
    and
    len(v2_train["frame_index"]) == expected_v2_train_frames
    and
    len(v2_train["timestamp"]) == expected_v2_train_frames
)

print("\nShape Check:")
print(pass_fail(shape_check))


# ------------------------------------------------------------
# 3. NaN / Inf Check
# ------------------------------------------------------------

numeric_keys = [
    "state",
    "action",
    "episode_index",
    "frame_index",
    "timestamp",
]

nan_inf_ok = True

for key in numeric_keys:

    arr = np.asarray(v2_train[key])

    if not np.all(np.isfinite(arr)):
        nan_inf_ok = False
        print(f"Invalid values found in: {key}")

print("\nNaN / Inf Check:")
print(pass_fail(nan_inf_ok))


# ------------------------------------------------------------
# 4. Original Train Preservation Check
# ------------------------------------------------------------

original_frames = len(v1_train["state"])

preservation_ok = True

for key in v1_train.files:

    arr1 = v1_train[key]
    arr2 = v2_train[key]

    if (
        hasattr(arr1, "shape")
        and arr1.ndim > 0
        and arr1.shape[0] == original_frames
    ):

        if not np.array_equal(
            arr1,
            arr2[:original_frames],
        ):
            preservation_ok = False
            print(
                f"Original preservation failed: {key}"
            )


print("\nOriginal Dataset v1 Preservation:")
print(pass_fail(preservation_ok))


# ------------------------------------------------------------
# 5. Read High-dynamics Source Frames
# ------------------------------------------------------------

high_dynamic_pairs = set()

with open(
    HIGH_DYNAMIC_REPORT,
    "r",
    encoding="utf-8-sig",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        pair = (
            int(row["episode_index"]),
            int(row["frame_index"]),
        )

        high_dynamic_pairs.add(pair)


print("\nExpected High-dynamics Frames:")
print(len(high_dynamic_pairs))


# ------------------------------------------------------------
# 6. Multiplicity Check
# ------------------------------------------------------------

v2_episode = np.asarray(
    v2_train["episode_index"]
)

v2_frame = np.asarray(
    v2_train["frame_index"]
)

frame_counts = {}

for ep, frame in zip(
    v2_episode,
    v2_frame,
):

    pair = (
        int(ep),
        int(frame),
    )

    frame_counts[pair] = (
        frame_counts.get(pair, 0)
        + 1
    )


high_dynamic_correct = 0
high_dynamic_wrong = 0

normal_correct = 0
normal_wrong = 0


for pair, count in frame_counts.items():

    if pair in high_dynamic_pairs:

        if count == 3:
            high_dynamic_correct += 1
        else:
            high_dynamic_wrong += 1

    else:

        if count == 1:
            normal_correct += 1
        else:
            normal_wrong += 1


oversampling_ok = (
    high_dynamic_correct
    == len(high_dynamic_pairs)
    and
    high_dynamic_wrong == 0
    and
    normal_wrong == 0
)


print("\nOversampling Check:")
print(pass_fail(oversampling_ok))

print("\nHigh-dynamics correct:")
print(high_dynamic_correct)

print("High-dynamics wrong:")
print(high_dynamic_wrong)

print("Normal correct:")
print(normal_correct)

print("Normal wrong:")
print(normal_wrong)


# ------------------------------------------------------------
# 7. Episode Start Duplication Check
# ------------------------------------------------------------

episode_start_ok = True

unique_episodes = np.unique(
    v1_train["episode_index"]
)

for ep in unique_episodes:

    mask = (
        v1_train["episode_index"] == ep
    )

    episode_frames = (
        v1_train["frame_index"][mask]
    )

    first_frame = int(
        np.min(episode_frames)
    )

    pair = (
        int(ep),
        first_frame,
    )

    if frame_counts.get(pair, 0) != 1:
        episode_start_ok = False
        print(
            f"Episode start duplicated: {pair}"
        )


print("\nEpisode Boundary Check:")
print(pass_fail(episode_start_ok))


# ------------------------------------------------------------
# 8. Validation unchanged
# ------------------------------------------------------------

val_same = True

for key in v1_val.files:

    if not np.array_equal(
        v1_val[key],
        v2_val[key],
    ):
        val_same = False
        print(
            f"Validation differs: {key}"
        )


print("\nValidation Unchanged:")
print(pass_fail(val_same))


# ------------------------------------------------------------
# 9. Test unchanged
# ------------------------------------------------------------

test_same = True

for key in v1_test.files:

    if not np.array_equal(
        v1_test[key],
        v2_test[key],
    ):
        test_same = False
        print(
            f"Test differs: {key}"
        )


print("\nTest Unchanged:")
print(pass_fail(test_same))


# ------------------------------------------------------------
# 10. Distribution Change
# ------------------------------------------------------------

original_high_ratio = (
    len(high_dynamic_pairs)
    / len(v1_train["state"])
)

v2_high_samples = (
    len(high_dynamic_pairs) * 3
)

v2_high_ratio = (
    v2_high_samples
    / len(v2_train["state"])
)


print("\nHigh-dynamics Sampling Ratio:")

print(
    "Dataset v1:",
    f"{original_high_ratio * 100:.2f}%"
)

print(
    "Dataset v2:",
    f"{v2_high_ratio * 100:.2f}%"
)


# ------------------------------------------------------------
# 11. Overall QC Result
# ------------------------------------------------------------

overall_pass = all(
    [
        shape_check,
        nan_inf_ok,
        preservation_ok,
        oversampling_ok,
        episode_start_ok,
        val_same,
        test_same,
    ]
)


print("\n" + "=" * 70)

print(
    "Overall Dataset v2 QC:",
    pass_fail(overall_pass),
)

print("=" * 70)


# ------------------------------------------------------------
# 12. Save Report
# ------------------------------------------------------------

rows = [
    ["check", "result"],

    [
        "shape_check",
        pass_fail(shape_check),
    ],

    [
        "nan_inf_check",
        pass_fail(nan_inf_ok),
    ],

    [
        "original_train_preservation",
        pass_fail(preservation_ok),
    ],

    [
        "oversampling_check",
        pass_fail(oversampling_ok),
    ],

    [
        "episode_boundary_check",
        pass_fail(episode_start_ok),
    ],

    [
        "validation_unchanged",
        pass_fail(val_same),
    ],

    [
        "test_unchanged",
        pass_fail(test_same),
    ],

    [
        "high_dynamic_unique_frames",
        len(high_dynamic_pairs),
    ],

    [
        "high_dynamic_correct_count",
        high_dynamic_correct,
    ],

    [
        "high_dynamic_wrong_count",
        high_dynamic_wrong,
    ],

    [
        "normal_wrong_count",
        normal_wrong,
    ],

    [
        "dataset_v1_high_dynamic_ratio_percent",
        original_high_ratio * 100,
    ],

    [
        "dataset_v2_high_dynamic_ratio_percent",
        v2_high_ratio * 100,
    ],

    [
        "overall_qc",
        pass_fail(overall_pass),
    ],
]


with open(
    SUMMARY_REPORT,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)
    writer.writerows(rows)


print("\nQC Report:")
print(SUMMARY_REPORT)