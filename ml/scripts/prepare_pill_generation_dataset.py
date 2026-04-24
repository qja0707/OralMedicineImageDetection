from pathlib import Path

from ml.src.data.prepare_pill_generation_dataset import prepare_pill_generation_dataset


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "ml" / "data"

    prepare_pill_generation_dataset(
        crop_metadata_path=data_root / "interim" / "pill_crops" / "crop_metadata.json",
        output_dir=data_root / "interim" / "pill_generation",
    )


if __name__ == "__main__":
    main()
