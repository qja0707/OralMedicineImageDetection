from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.augment.classification_augment import balance_train_split


def main():
    data_root = project_root / "ml" / "data"

    stats = balance_train_split(
        input_dir=data_root / "processed" / "classify" / "mixed",
        output_dir=data_root / "processed" / "classify" / "mixed_aug",
        target_count=None,
        seed=42,
        keep_original=True,
        enable_horizontal_flip=False,
        enable_vertical_flip=False,
        balance_sources=True,
        source_names=("kaggle", "internet"),
    )
    print("=" * 60)
    print("YOLO classification crop augmentation complete")
    print("=" * 60)
    print(f"Input train      : {stats['input_train']}")
    print(f"Copied train     : {stats['copied_train']}")
    print(f"Copied val       : {stats['copied_val']}")
    print(f"Target per class : {stats['target_count']}")
    print(f"Balance sources  : {stats['balance_sources']}")
    print(f"Augmented train  : {stats['augmented']}")
    print(f"Skipped          : {stats['skipped']}")
    print(f"Output dir       : {stats['output_dir']}")
    print(f"Balance report   : {stats['report_path']}")


if __name__ == "__main__":
    main()
