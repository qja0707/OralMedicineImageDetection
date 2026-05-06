from pathlib import Path
import runpy
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def ensure_dataset():
    dataset_dir = project_root / "ml" / "data" / "processed" / "yolo_single_class"
    yaml_path = dataset_dir / "pill.yaml"
    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_dir / "labels" / "val").glob("*.txt"))
    if yaml_path.exists() and train_labels and val_labels:
        return yaml_path

    runpy.run_path(
        str(project_root / "ml" / "scripts" / "2step_train" / "05_build_kaggle_single_class_yolo_dataset.py"),
        run_name="__main__",
    )
    return yaml_path


def train_yolo11l_single_class_detector():
    from ultralytics import YOLO
    import torch

    data_yaml_path = ensure_dataset()
    model = YOLO("yolo11l.pt")

    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "patience": 25,
        "workers": 8,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "pretrained": True,
        "project": str(project_root / "ml" / "outputs" / "checkpoints"),
        "name": "yolo11l_kaggle_single_class_detector",
        "exist_ok": True,
        "verbose": True,
        "deterministic": False,
        "mosaic": 1.0,
        "degrees": 180.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "scale": 0.3,
        "translate": 0.1,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "copy_paste": 0.0,
    }
    results = model.train(**train_kwargs)

    print("=" * 60)
    print("YOLO11L single-class detector 학습 완료")
    print("=" * 60)
    print(f"  Dataset YAML       : {data_yaml_path}")
    print(f"  초기 가중치        : yolo11l.pt")
    print(f"  사용 device        : {train_kwargs['device']}")
    print(f"  학습 결과 디렉터리 : {results.save_dir}")
    print(f"  best.pt            : {results.save_dir / 'weights' / 'best.pt'}")
    print(f"  last.pt            : {results.save_dir / 'weights' / 'last.pt'}")

    return results


def main():
    train_yolo11l_single_class_detector()


if __name__ == "__main__":
    main()
