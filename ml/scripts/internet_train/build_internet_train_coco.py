"""Build the initial internet-train COCO files from raw internet images.

This script runs automatic bbox detection and writes internet_train_coco.json,
internet_train_bbox_metadata.json, and bbox preview images. Do not run it after
manual bbox correction unless you intentionally want to regenerate and overwrite
the corrected JSON files.
"""

from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.data.build_internet_train_coco import build_internet_train_coco


def main():
    data_root = project_root / "ml" / "data"
    raw_internet_root = data_root / "raw" / "internet_train"

    build_internet_train_coco(
        internet_images_dir=raw_internet_root / "images",
        mapping_csv_path=raw_internet_root / "metadata" / "internet_image_mapping.csv",
        output_dir=data_root / "interim" / "internet_train",
    )


if __name__ == "__main__":
    main()
