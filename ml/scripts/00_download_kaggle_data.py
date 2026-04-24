import argparse
import os
import shutil
from pathlib import Path

import kagglehub


COMPETITION_NAME = "ai10-level1-project"


def load_dotenv(dotenv_path: Path) -> dict[str, str]:
    values = {}

    if not dotenv_path.exists():
        raise FileNotFoundError(f".env file not found: {dotenv_path}")

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        values[key] = value

    return values


def find_required_dir(download_root: Path, target_name: str) -> Path:
    candidates = sorted(
        path for path in download_root.rglob(target_name) if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find `{target_name}` under {download_root}")
    return candidates[0]


def copy_tree(source_dir: Path, target_dir: Path) -> int:
    copied = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    for source_path in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Download Kaggle competition data into ml/data/raw."
    )
    parser.add_argument(
        "--competition",
        default=COMPETITION_NAME,
        help="Kaggle competition slug.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / "ml" / ".env"
    raw_root = project_root / "ml" / "data" / "raw"
    raw_images_dir = raw_root / "images"
    raw_annotations_dir = raw_root / "annotations"

    env_values = load_dotenv(dotenv_path)
    kaggle_api_token = env_values.get("KAGGLE_API_TOKEN")
    if not kaggle_api_token:
        raise EnvironmentError(f"KAGGLE_API_TOKEN is missing in {dotenv_path}")

    os.environ["KAGGLE_API_TOKEN"] = kaggle_api_token

    print("=" * 55)
    print("1단계: Kaggle competition 데이터 다운로드 중...")
    print("=" * 55)
    download_path = Path(kagglehub.competition_download(args.competition))
    print(f"  다운로드 경로      : {download_path}")

    train_images_dir = find_required_dir(download_path, "train_images")
    train_annotations_dir = find_required_dir(download_path, "train_annotations")

    print("\n" + "=" * 55)
    print("2단계: raw 데이터 디렉터리로 복사 중...")
    print("=" * 55)
    copied_images = copy_tree(train_images_dir, raw_images_dir)
    copied_annotations = copy_tree(train_annotations_dir, raw_annotations_dir)
    print(f"  이미지 복사 수      : {copied_images}개")
    print(f"  어노테이션 복사 수  : {copied_annotations}개")

    print("\n" + "=" * 55)
    print("  완료!")
    print("=" * 55)
    print(f"  raw 이미지 경로     : {raw_images_dir}")
    print(f"  raw 어노테이션 경로 : {raw_annotations_dir}")


if __name__ == "__main__":
    main()
