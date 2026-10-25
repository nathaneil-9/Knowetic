import random
from pathlib import Path

SOURCE_DIR = Path("data/processed/faces")
OUTPUT_DIR = Path("data/processed/split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

for label in ["real", "fake"]:
    source_label_dir = SOURCE_DIR / label

    video_dirs = [
        d for d in source_label_dir.iterdir()
        if d.is_dir()
    ]

    random.shuffle(video_dirs)

    total = len(video_dirs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": video_dirs[:train_end],
        "val": video_dirs[train_end:val_end],
        "test": video_dirs[val_end:]
    }

    for split_name, split_videos in splits.items():

        output_label_dir = OUTPUT_DIR / split_name / label
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for video_dir in split_videos:

            output_video_dir = output_label_dir / video_dir.name
            output_video_dir.mkdir(parents=True, exist_ok=True)

            for image_path in video_dir.glob("*.jpg"):
                link_path = output_video_dir / image_path.name

                if not link_path.exists():
                    link_path.symlink_to(image_path.resolve())

        print(
            f"{label} | {split_name}: "
            f"{len(split_videos)} videos"
        )

print("\nDataset split complete.")
