from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.data.build_yolo_classification_dataset import (
    build_yolo_classification_dataset_from_sources,
)


def main():
    data_root = project_root / "ml" / "data"

    sources = [
        {
            "name": "kaggle",
            "coco_path": data_root / "interim" / "merged" / "train_coco.json",
            "raw_images_dir": data_root / "raw" / "images",
        },
        {
            "name": "internet",
            "coco_path": data_root / "interim" / "internet_train" / "internet_train_coco.json",
            "raw_images_dir": data_root / "raw" / "internet_train" / "images",
        },
    ]

    build_yolo_classification_dataset_from_sources(
        sources=sources,
        output_dir=data_root / "processed" / "classify" / "mixed",
        val_ratio=0.2,
        seed=42,
        skip_missing=True,
    )


if __name__ == "__main__":
    main()
