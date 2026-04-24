from pathlib import Path

from ml.src.data.merge_coco import build_merged_coco


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "ml" / "data"

    build_merged_coco(
        train_annotations_path=data_root / "raw" / "annotations",
        train_images_path=data_root / "raw" / "images",
        output_dir=data_root / "interim" / "merged",
    )


if __name__ == "__main__":
    main()
