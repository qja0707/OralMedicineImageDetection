from pathlib import Path

from ml.src.data.build_pill_crops import build_pill_crops


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "ml" / "data"

    build_pill_crops(
        coco_json_path=data_root / "interim" / "merged" / "train_coco.json",
        annotation_metadata_path=data_root / "interim" / "metadata" / "annotation_metadata.json",
        raw_images_dir=data_root / "raw" / "images",
        output_dir=data_root / "interim" / "pill_crops",
        margin_ratio=0.1,
    )


if __name__ == "__main__":
    main()
