from pathlib import Path
import argparse
import runpy
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.train.train_yolo_classification import train_yolo_classification


def ensure_classification_dataset(data_root):
    dataset_dir = data_root / "processed" / "classify" / "mixed_aug"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if train_dir.exists() and val_dir.exists() and list(train_dir.glob("*/*")):
        return dataset_dir

    runpy.run_path(
        str(project_root / "ml" / "scripts" / "2step_train" / "01_build_yolo_classification_dataset.py"),
        run_name="__main__",
    )
    runpy.run_path(
        str(project_root / "ml" / "scripts" / "2step_train" / "02_augment_yolo_classification_dataset.py"),
        run_name="__main__",
    )
    return dataset_dir


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11L classification model on mixed crop dataset.")
    parser.add_argument("--model", default="yolo11l-cls.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="yolo11l_cls_mixed_aug")
    parser.add_argument("--project", default="ml/outputs/checkpoints")
    parser.add_argument("--exist-ok", action="store_true", default=True)
    args = parser.parse_args()

    data_root = project_root / "ml" / "data"
    dataset_dir = ensure_classification_dataset(data_root)

    train_yolo_classification(
        data_dir=dataset_dir,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
