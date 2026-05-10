from pathlib import Path
import argparse
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.inference.double_step_inference import double_step_inference


def main():
    parser = argparse.ArgumentParser(description="first step: find boxes")
    parser.add_argument(
        "--detect_model",
        default=str(project_root / "ml" / "outputs" / "checkpoints" / "yolo11l_kaggle_single_class_detector" / "weights" / "best.pt"),
    )
    parser.add_argument(
        "--classify_model",
        default=str(project_root / "ml" / "outputs" / "checkpoints" / "yolo11l_cls_mixed_aug" / "weights" / "best.pt"),
    )
    parser.add_argument(
        "--images",
        default=str(project_root / "ml" / "data" / "raw" / "test_images"),
    )
    parser.add_argument(
        "--output_path",
        default=str(project_root / "ml" / "outputs" / "logs" / "submission.csv"),
    )

    args = parser.parse_args()

    double_step_inference(
        detect_model_path=args.detect_model,
        classify_model_path=args.classify_model,
        image_path=args.images,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
