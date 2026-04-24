from pathlib import Path

from ml.src.data.prepare_sdxl_lora_dataset import prepare_sdxl_lora_dataset


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "ml" / "data"

    prepare_sdxl_lora_dataset(
        metadata_path=data_root / "interim" / "pill_generation" / "metadata.json",
        crop_root_dir=data_root / "interim" / "pill_crops",
        output_dir=data_root / "interim" / "sdxl_lora",
    )


if __name__ == "__main__":
    main()
