from pathlib import Path
import runpy
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.data.build_yolo_dataset import build_yolo_dataset


def ensure_kaggle_coco(data_root):
    merged_coco_path = data_root / "interim" / "merged" / "train_coco.json"
    train_coco_path = data_root / "interim" / "splits" / "train_coco.json"
    val_coco_path = data_root / "interim" / "splits" / "val_coco.json"

    if not merged_coco_path.exists():
        runpy.run_path(
            str(project_root / "ml" / "scripts" / "01_merge_coco.py"),
            run_name="__main__",
        )
    if not train_coco_path.exists() or not val_coco_path.exists():
        runpy.run_path(
            str(project_root / "ml" / "scripts" / "02_split_coco.py"),
            run_name="__main__",
        )

    return train_coco_path, val_coco_path


def main():
    data_root = project_root / "ml" / "data"
    train_coco_path, val_coco_path = ensure_kaggle_coco(data_root)

    build_yolo_dataset(
        train_coco_path=train_coco_path,
        val_coco_path=val_coco_path,
        train_images_dir=data_root / "raw" / "images",
        val_images_dir=data_root / "raw" / "images",
        output_dir=data_root / "processed" / "yolo_single_class",
        single_class_name="pill",
        target_size=640,
    )


if __name__ == "__main__":
    main()
