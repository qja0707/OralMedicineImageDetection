from pathlib import Path

from ml.src.data.split_coco import split_coco_dataset


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "ml" / "data"

    split_coco_dataset(
        coco_json_path=data_root / "interim" / "merged" / "train_coco.json",
        output_dir=data_root / "interim" / "splits",
        seed=42,
        val_ratio=0.2,
    )


if __name__ == "__main__":
    main()
