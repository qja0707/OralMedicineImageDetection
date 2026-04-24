from pathlib import Path

from ml.src.train.train_sdxl_lora import train_sdxl_lora


def main():
    project_root = Path(__file__).resolve().parents[2]
    configs_root = project_root / "ml" / "configs"

    train_sdxl_lora(
        train_config_path=configs_root / "train" / "sdxl_lora.yaml",
    )


if __name__ == "__main__":
    main()
