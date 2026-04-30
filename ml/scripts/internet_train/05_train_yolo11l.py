"""Train YOLO11L on internet-train data and original-eval validation data.

This uses the same Ultralytics YOLO txt dataset layout as the RT-DETR
experiment, but writes to YOLO11L-specific dataset and checkpoint directories
so existing RT-DETR outputs are not overwritten.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.augment.yolo_augment import get_yolo_train_augmentation
from ml.src.data.build_yolo_dataset import build_yolo_dataset


COPY_PASTE_AUGMENT_FACTOR = 20
DATASET_NAME = "internet_train_yolo11l"
RUN_NAME = "internet_train_yolo11l_multi_object"
EXPECTED_COPY_PASTE_METADATA = {
    "augment_factor": COPY_PASTE_AUGMENT_FACTOR,
    "object_count_probs": {"2": 0.2, "3": 0.3, "4": 0.5},
    "same_class_duplicate": False,
    "color_augmentation": False,
}


def load_rtdetr_dataset_builder():
    builder_path = project_root / "ml" / "scripts" / "internet_train" / "03_build_rtdetr_dataset.py"
    spec = spec_from_file_location("internet_train_rtdetr_dataset_builder", builder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load dataset builder: {builder_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_expected_copy_paste_metadata(dataset_dir):
    metadata_path = dataset_dir / "copy_paste_metadata.json"
    if not metadata_path.exists():
        return False
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    for key, expected_value in EXPECTED_COPY_PASTE_METADATA.items():
        if metadata.get(key) != expected_value:
            return False
    return True


def ensure_dataset(data_root):
    dataset_dir = data_root / "processed" / DATASET_NAME
    yaml_path = dataset_dir / "pill.yaml"
    copy_paste_labels = list((dataset_dir / "labels" / "train").glob("copy_paste_*.txt"))
    train_coco_path = data_root / "interim" / "internet_train" / "internet_train_coco.json"

    with open(train_coco_path, "r", encoding="utf-8") as file:
        train_coco = json.load(file)
    expected_copy_paste_count = len(train_coco["images"]) * COPY_PASTE_AUGMENT_FACTOR

    if (
        yaml_path.exists()
        and len(copy_paste_labels) >= expected_copy_paste_count
        and has_expected_copy_paste_metadata(dataset_dir)
    ):
        return yaml_path

    train_images_dir = data_root / "raw" / "internet_train" / "images"
    build_yolo_dataset(
        train_coco_path=train_coco_path,
        val_coco_path=data_root / "interim" / "internet_train" / "original_eval_coco.json",
        train_images_dir=train_images_dir,
        val_images_dir=data_root / "raw" / "images",
        output_dir=dataset_dir,
    )

    builder = load_rtdetr_dataset_builder()
    builder.add_copy_paste_train_samples(
        coco_path=train_coco_path,
        images_dir=train_images_dir,
        dataset_dir=dataset_dir,
        augment_factor=COPY_PASTE_AUGMENT_FACTOR,
    )
    return yaml_path


def train_yolo11l():
    from ultralytics import YOLO
    import torch

    data_root = project_root / "ml" / "data"
    configs_root = project_root / "ml" / "configs"
    data_yaml_path = ensure_dataset(data_root)
    previous_best_path = (
        project_root
        / "ml"
        / "outputs"
        / "checkpoints"
        / RUN_NAME
        / "weights"
        / "best.pt"
    )
    model_path = previous_best_path if previous_best_path.exists() else "yolo11l.pt"

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
        "name": RUN_NAME,
        "exist_ok": False,
        "verbose": True,
        "deterministic": False,
        **augmentation,
    }

    model = YOLO(str(model_path))
    results = model.train(**train_kwargs)

    print("=" * 55)
    print("인터넷 train YOLO11L 학습 완료")
    print("=" * 55)
    print(f"  Dataset YAML       : {data_yaml_path}")
    print(f"  초기 가중치        : {model_path}")
    print(f"  사용 device        : {train_kwargs['device']}")
    print(f"  학습 결과 디렉터리 : {results.save_dir}")
    print(f"  best.pt            : {results.save_dir / 'weights' / 'best.pt'}")
    print(f"  last.pt            : {results.save_dir / 'weights' / 'last.pt'}")

    return results


def main():
    train_yolo11l()


if __name__ == "__main__":
    main()
