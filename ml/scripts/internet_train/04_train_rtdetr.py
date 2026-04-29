"""Train RT-DETR on internet-train data and original-eval validation data.

If the Ultralytics dataset directory is missing, this script builds it first
from the prepared COCO files.
"""

from pathlib import Path
import json
import runpy
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.augment.yolo_augment import get_yolo_train_augmentation


COPY_PASTE_AUGMENT_FACTOR = 20


def ensure_dataset(data_root):
    dataset_dir = data_root / "processed" / "internet_train_rtdetr"
    yaml_path = dataset_dir / "pill.yaml"
    copy_paste_labels = list((dataset_dir / "labels" / "train").glob("copy_paste_*.txt"))
    train_coco_path = data_root / "interim" / "internet_train" / "internet_train_coco.json"
    with open(train_coco_path, "r", encoding="utf-8") as file:
        train_coco = json.load(file)
    expected_copy_paste_count = len(train_coco["images"]) * COPY_PASTE_AUGMENT_FACTOR
    if yaml_path.exists() and len(copy_paste_labels) >= expected_copy_paste_count:
        return yaml_path

    runpy.run_path(
        str(project_root / "ml" / "scripts" / "internet_train" / "03_build_rtdetr_dataset.py"),
        run_name="__main__",
    )
    return yaml_path


def train_rtdetr():
    from ultralytics import RTDETR
    import torch

    data_root = project_root / "ml" / "data"
    configs_root = project_root / "ml" / "configs"
    data_yaml_path = ensure_dataset(data_root)

    augmentation = get_yolo_train_augmentation(configs_root / "augment.yaml")
    augmentation.update(
        {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "copy_paste": 0.0,
        }
    )
    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": 150,
        "batch": 16,
        "patience": 40,
        "workers": 8,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "pretrained": True,
        "project": str(project_root / "ml" / "outputs" / "checkpoints"),
        "name": "internet_train_rtdetr_l",
        "exist_ok": True,
        "verbose": True,
        "deterministic": False,
        **augmentation,
    }

    model = RTDETR("rtdetr-l.pt")
    results = model.train(**train_kwargs)

    print("=" * 55)
    print("인터넷 train RT-DETR 학습 완료")
    print("=" * 55)
    print(f"  Dataset YAML       : {data_yaml_path}")
    print(f"  사용 device        : {train_kwargs['device']}")
    print(f"  학습 결과 디렉터리 : {results.save_dir}")
    print(f"  best.pt            : {results.save_dir / 'weights' / 'best.pt'}")
    print(f"  last.pt            : {results.save_dir / 'weights' / 'last.pt'}")

    return results


def main():
    train_rtdetr()


if __name__ == "__main__":
    main()
