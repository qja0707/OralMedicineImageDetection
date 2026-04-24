from pathlib import Path

from ml.src.train.train_yolo import train_yolo


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "ml" / "data"
    configs_root = project_root / "ml" / "configs"

    train_yolo(
        data_yaml_path=data_root / "processed" / "yolo" / "pill.yaml",
        augment_config_path=configs_root / "augment.yaml",
        train_config_path=configs_root / "train" / "yolo.yaml",
    )


if __name__ == "__main__":
    main()
