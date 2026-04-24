from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_AUGMENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "augment.yaml"
)


def load_augment_config(config_path=None):
    config_path = Path(config_path) if config_path else DEFAULT_AUGMENT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Augment config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_yolo_train_augmentation(config_path=None, overrides=None):
    config = load_augment_config(config_path)
    augmentation = deepcopy(config["yolo"]["train"])
    if overrides:
        augmentation.update(overrides)
    return augmentation


def get_yolo_val_settings(config_path=None, overrides=None):
    config = load_augment_config(config_path)
    settings = deepcopy(config["yolo"]["val"])
    if overrides:
        settings.update(overrides)
    return settings
