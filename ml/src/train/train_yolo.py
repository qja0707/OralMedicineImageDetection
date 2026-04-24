from copy import deepcopy
from pathlib import Path
import shutil

import yaml

from ml.src.augment.yolo_augment import get_yolo_train_augmentation


DEFAULT_TRAIN_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "train" / "yolo.yaml"
)


def load_train_config(config_path=None):
    config_path = Path(config_path) if config_path else DEFAULT_TRAIN_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Train config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def copy_training_artifacts(save_dir, project_root):
    save_dir = Path(save_dir)
    models_yolo_dir = Path(project_root) / "models" / "yolo"
    models_yolo_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = save_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"
    best_dst = models_yolo_dir / "best.pt"
    last_dst = models_yolo_dir / "last.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
    if last_src.exists():
        shutil.copy2(last_src, last_dst)

    return {
        "save_dir": save_dir,
        "weights_dir": weights_dir,
        "best_src": best_src,
        "last_src": last_src,
        "best_dst": best_dst,
        "last_dst": last_dst,
        "results_csv": save_dir / "results.csv",
        "results_png": save_dir / "results.png",
        "confusion_matrix": save_dir / "confusion_matrix.png",
    }


def train_yolo(
    data_yaml_path,
    augment_config_path=None,
    train_config_path=None,
    overrides=None,
):
    from ultralytics import YOLO

    project_root = Path(__file__).resolve().parents[3]
    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YOLO dataset yaml not found: {data_yaml_path}")

    train_config = deepcopy(load_train_config(train_config_path)["yolo"])
    augmentation = get_yolo_train_augmentation(augment_config_path)

    if overrides:
        train_config.update(overrides)

    model_name = train_config.pop("model")
    model = YOLO(model_name)

    train_kwargs = {
        "data": str(data_yaml_path),
        **train_config,
        **augmentation,
    }
    device = train_kwargs.get("device")
    if device is None:
        train_kwargs.pop("device", None)

    results = model.train(**train_kwargs)
    artifacts = copy_training_artifacts(results.save_dir, project_root)

    print("=" * 55)
    print("YOLO 학습 완료")
    print("=" * 55)
    print(f"  학습 결과 디렉터리 : {artifacts['save_dir']}")
    print(f"  best.pt 원본       : {artifacts['best_src']}")
    print(f"  last.pt 원본       : {artifacts['last_src']}")
    print(f"  best.pt 복사본     : {artifacts['best_dst']}")
    print(f"  last.pt 복사본     : {artifacts['last_dst']}")
    print(f"  results.csv        : {artifacts['results_csv']}")
    print(f"  results.png        : {artifacts['results_png']}")
    print(f"  confusion_matrix   : {artifacts['confusion_matrix']}")

    return {
        "results": results,
        "artifacts": artifacts,
    }
