from pathlib import Path

from ml.src.data.build_yolo_dataset import build_yolo_dataset


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "ml" / "data"

    build_yolo_dataset(
        train_coco_path=data_root / "interim" / "splits" / "train_coco.json",
        val_coco_path=data_root / "interim" / "splits" / "val_coco.json",
        raw_images_dir=data_root / "raw" / "images",
        output_dir=data_root / "processed" / "yolo",
    )


if __name__ == "__main__":
    main()
