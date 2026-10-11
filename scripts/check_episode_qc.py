from pathlib import Path
import csv
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = PROJECT_ROOT / "raw_data" / "episodes"
METADATA_FILE = PROJECT_ROOT / "metadata" / "episode_metadata.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "episode_qc_report.csv"

results = []

# =========================
# Read metadata
# =========================

with METADATA_FILE.open(
    "r",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)
    metadata = list(reader)

print("=" * 65)
print("Episode Technical QC")
print("=" * 65)

print(f"Metadata Episodes: {len(metadata)}")
print()

# =========================
# Check each episode
# =========================

for row in metadata:

    episode_id = row["episode_id"]
    video_file = row["video_file"]

    video_path = VIDEO_DIR / video_file

    exists = video_path.exists()

    readable = False
    fps = 0
    frame_count = 0
    width = 0
    height = 0
    duration = 0

    if exists:

        cap = cv2.VideoCapture(str(video_path))

        readable = cap.isOpened()

        if readable:

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            if fps > 0:
                duration = frame_count / fps

        cap.release()

    # Basic QC
    status = "PASS"

    if not exists:
        status = "FAIL_MISSING"

    elif not readable:
        status = "FAIL_UNREADABLE"

    elif frame_count <= 0:
        status = "FAIL_NO_FRAMES"

    elif duration < 3:
        status = "REVIEW_TOO_SHORT"

    elif duration > 20:
        status = "REVIEW_TOO_LONG"

    results.append({
        "episode_id": episode_id,
        "video_file": video_file,
        "exists": exists,
        "readable": readable,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 2),
        "qc_status": status,
    })

    print(
        f"{episode_id} | "
        f"{duration:.2f}s | "
        f"{fps:.2f} FPS | "
        f"{width}x{height} | "
        f"{status}"
    )

# =========================
# Save report
# =========================

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

fieldnames = [
    "episode_id",
    "video_file",
    "exists",
    "readable",
    "fps",
    "frame_count",
    "width",
    "height",
    "duration_sec",
    "qc_status",
]

with REPORT_FILE.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

# =========================
# Summary
# =========================

pass_count = sum(
    r["qc_status"] == "PASS"
    for r in results
)

review_count = sum(
    r["qc_status"].startswith("REVIEW")
    for r in results
)

fail_count = sum(
    r["qc_status"].startswith("FAIL")
    for r in results
)

print()
print("=" * 65)
print("QC Summary")
print("=" * 65)

print(f"PASS   : {pass_count}")
print(f"REVIEW : {review_count}")
print(f"FAIL   : {fail_count}")

print()
print(f"报告：{REPORT_FILE}")

print()
print("Episode Technical QC 完成 ✅")
