"""Train RT-DETR on internet-train data and original-eval validation data.

If the Ultralytics dataset directory is missing, this script builds it first
from the prepared COCO files.
"""

from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.augment.yolo_augment import get_yolo_train_augmentation
from ml.src.data.build_yolo_dataset import build_yolo_dataset


def ensure_dataset(data_root):
    dataset_dir = data_root / "processed" / "internet_train_rtdetr"
    yaml_path = dataset_dir / "pill.yaml"
    if yaml_path.exists():
        return yaml_path

    build_yolo_dataset(
        train_coco_path=data_root / "interim" / "internet_train" / "internet_train_coco.json",
        val_coco_path=data_root / "interim" / "internet_train" / "original_eval_coco.json",
        train_images_dir=data_root / "raw" / "internet_train" / "images",
        val_images_dir=data_root / "raw" / "images",
        output_dir=dataset_dir,
    )
    return yaml_path


def train_rtdetr():
    from ultralytics import RTDETR
    import torch

    data_root = project_root / "ml" / "data"
    configs_root = project_root / "ml" / "configs"
    data_yaml_path = ensure_dataset(data_root)

    augmentation = get_yolo_train_augmentation(configs_root / "augment.yaml")
    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": 100,
        "batch": 16,
        "patience": 20,
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
