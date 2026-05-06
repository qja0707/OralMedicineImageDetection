from pathlib import Path

from ml.src.data.split_coco import split_coco_dataset
import sys

def main():
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    data_root = project_root / "ml" / "data"

    split_coco_dataset(
        coco_json_path=data_root / "interim" / "merged" / "train_coco.json",
        output_dir=data_root / "interim" / "splits",
        seed=42,
        val_ratio=0.2,
        force_single_group_classes_to_val=False
    )


if __name__ == "__main__":
    main()
