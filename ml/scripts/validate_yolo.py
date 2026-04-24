from pathlib import Path

from ml.src.train.validate_yolo import plot_low_per_class_metrics


def main():
    project_root = Path(__file__).resolve().parents[2]
    plot_low_per_class_metrics(
        weights_path=project_root / "models" / "yolo" / "best.pt",
        data_yaml_path=project_root / "ml" / "data" / "processed" / "yolo" / "pill.yaml",
        top_k=10,
        output_path=project_root / "ml" / "outputs" / "logs" / "yolo_validation" / "low_per_class_metrics.png",
    )


if __name__ == "__main__":
    main()
