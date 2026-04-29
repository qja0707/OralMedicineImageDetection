import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"COCO json not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_metadata(coco: dict, source_coco_path: Path, eval_coco_path: Path) -> dict:
    category_ids = {category["id"] for category in coco.get("categories", [])}
    annotated_category_ids = {
        annotation["category_id"] for annotation in coco.get("annotations", [])
    }

    return {
        "experiment": "internet_train_original_eval",
        "description": (
            "Use the full original captured dataset as a fixed evaluation set. "
            "Training data will be built separately from internet pill images."
        ),
        "source_coco_path": str(source_coco_path),
        "eval_coco_path": str(eval_coco_path),
        "num_images": len(coco.get("images", [])),
        "num_annotations": len(coco.get("annotations", [])),
        "num_categories": len(coco.get("categories", [])),
        "num_annotated_categories": len(annotated_category_ids),
        "category_ids_without_annotations": sorted(category_ids - annotated_category_ids),
    }


def prepare_original_eval(source_coco_path: Path, output_dir: Path) -> dict:
    coco = load_json(source_coco_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_coco_path = output_dir / "original_eval_coco.json"
    metadata_path = output_dir / "original_eval_metadata.json"

    shutil.copy2(source_coco_path, eval_coco_path)

    metadata = build_metadata(
        coco=coco,
        source_coco_path=source_coco_path,
        eval_coco_path=eval_coco_path,
    )
    write_json(metadata_path, metadata)

    return {
        "eval_coco_path": eval_coco_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def main():
    project_root = Path(__file__).resolve().parents[3]
    data_root = project_root / "ml" / "data"

    parser = argparse.ArgumentParser(
        description="Prepare the full original COCO dataset as eval data."
    )
    parser.add_argument(
        "--source-coco-path",
        type=Path,
        default=data_root / "interim" / "merged" / "train_coco.json",
        help="Merged original COCO json to use as the fixed eval set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "interim" / "internet_eval",
        help="Directory for internet-eval experiment artifacts.",
    )
    args = parser.parse_args()

    result = prepare_original_eval(
        source_coco_path=args.source_coco_path,
        output_dir=args.output_dir,
    )
    metadata = result["metadata"]

    print("=" * 55)
    print("인터넷 train 실험용 original eval 세트 준비 완료")
    print("=" * 55)
    print(f"  입력 COCO          : {args.source_coco_path}")
    print(f"  Eval COCO          : {result['eval_coco_path']}")
    print(f"  Metadata           : {result['metadata_path']}")
    print(f"  이미지 수          : {metadata['num_images']}개")
    print(f"  어노테이션 수      : {metadata['num_annotations']}개")
    print(f"  클래스 수          : {metadata['num_categories']}개")
    print(f"  어노테이션 클래스 수: {metadata['num_annotated_categories']}개")


if __name__ == "__main__":
    main()
