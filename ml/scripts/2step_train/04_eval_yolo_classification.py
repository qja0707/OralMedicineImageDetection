from pathlib import Path
import argparse
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.train.evaluate_yolo_classification import evaluate_yolo_classification


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO classification model.")
    parser.add_argument(
        "--model",
        default=str(project_root / "models" / "yolo_cls" / "best.pt"),
    )
    parser.add_argument(
        "--data-dir",
        default=str(project_root / "ml" / "data" / "processed" / "classify" / "mixed_aug"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "ml" / "outputs" / "classification_eval"),
    )
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    evaluate_yolo_classification(
        model_path=args.model,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        device=args.device,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
