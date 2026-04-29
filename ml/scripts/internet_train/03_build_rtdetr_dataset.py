"""Build the RT-DETR/Ultralytics dataset for the internet-train experiment.

Train uses manually corrected internet images. Validation uses the original
captured dataset prepared by 01_prepare_original_eval.py.
"""

from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.data.build_yolo_dataset import build_yolo_dataset


def main():
    data_root = project_root / "ml" / "data"

    build_yolo_dataset(
        train_coco_path=data_root / "interim" / "internet_train" / "internet_train_coco.json",
        val_coco_path=data_root / "interim" / "internet_train" / "original_eval_coco.json",
        train_images_dir=data_root / "raw" / "internet_train" / "images",
        val_images_dir=data_root / "raw" / "images",
        output_dir=data_root / "processed" / "internet_train_rtdetr",
    )


if __name__ == "__main__":
    main()
