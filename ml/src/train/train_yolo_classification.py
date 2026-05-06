from pathlib import Path
import shutil


def copy_classification_artifacts(save_dir, project_root):
    save_dir = Path(save_dir)
    models_dir = Path(project_root) / "models" / "yolo_cls"
    models_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = save_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"
    best_dst = models_dir / "best.pt"
    last_dst = models_dir / "last.pt"

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


def train_yolo_classification(
    data_dir,
    model="yolo11l-cls.pt",
    epochs=50,
    imgsz=224,
    batch=16,
    patience=15,
    workers=8,
    device=None,
    project="ml/outputs/checkpoints",
    name="yolo11l_cls_mixed_aug",
    exist_ok=True,
    verbose=True,
    deterministic=False,
    overrides=None,
):
    from ultralytics import YOLO
    import torch

    project_root = Path(__file__).resolve().parents[3]
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Classification dataset not found: {data_dir}")
    if not (data_dir / "train").exists():
        raise FileNotFoundError(f"Train split not found: {data_dir / 'train'}")
    if not (data_dir / "val").exists():
        raise FileNotFoundError(f"Val split not found: {data_dir / 'val'}")

    train_kwargs = {
        "data": str(data_dir),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "workers": workers,
        "device": device,
        "project": str((project_root / project).resolve()),
        "name": name,
        "exist_ok": exist_ok,
        "verbose": verbose,
        "deterministic": deterministic,
    }
    if train_kwargs["device"] is None:
        train_kwargs["device"] = 0 if torch.cuda.is_available() else "cpu"
    if overrides:
        train_kwargs.update(overrides)

    yolo_model = YOLO(str(model))
    results = yolo_model.train(**train_kwargs)
    artifacts = copy_classification_artifacts(results.save_dir, project_root)

    print("=" * 60)
    print("YOLO classification 학습 완료")
    print("=" * 60)
    print(f"  Dataset           : {data_dir}")
    print(f"  초기 가중치        : {model}")
    print(f"  사용 device        : {train_kwargs['device']}")
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
        "train_kwargs": train_kwargs,
    }
