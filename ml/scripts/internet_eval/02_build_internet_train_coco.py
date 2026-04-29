from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.data.build_internet_train_coco import build_internet_train_coco


def main():
    data_root = project_root / "ml" / "data"
    raw_internet_root = data_root / "raw" / "internet_eval"

    build_internet_train_coco(
        internet_images_dir=raw_internet_root / "images",
        mapping_csv_path=raw_internet_root / "metadata" / "internet_image_mapping.csv",
        output_dir=data_root / "interim" / "internet_eval",
    )


if __name__ == "__main__":
    main()
