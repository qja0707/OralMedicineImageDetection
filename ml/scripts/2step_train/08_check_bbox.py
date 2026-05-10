from pathlib import Path
import argparse
import sys


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.utils.draw_bbox_from_submission import draw_bbox_from_submission


def main():
    parser = argparse.ArgumentParser(description="draw bboxes")

    parser.add_argument(
        "--test_dir",
        default=str(project_root / "ml" / "data" / "raw" / "test_images"),
    )
    parser.add_argument(
        "--submission_path",
        default=str(project_root / "ml" / "outputs" / "logs" / "submission.csv"),
    )

    args = parser.parse_args()

    draw_bbox_from_submission(
        test_dir=args.test_dir,
        submission_path=args.submission_path,
    )


if __name__ == "__main__":
    main()
