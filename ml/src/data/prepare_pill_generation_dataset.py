import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


GENERATION_COLUMNS = [
    "annotation_id",
    "category_id",
    "crop_relative_path",
    "color_class1",
    "color_class2",
    "drug_shape",
    "print_front",
    "print_back",
    "line_front",
    "line_back",
]


def load_json(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def normalize_metadata_row(row):
    normalized = {
        "annotation_id": row["annotation_id"],
        "category_id": str(row["category_id"]),
        "crop_relative_path": normalize_text(row.get("crop_relative_path")),
        "color_class1": normalize_text(row.get("color_class1")),
        "color_class2": normalize_text(row.get("color_class2")),
        "drug_shape": normalize_text(row.get("drug_shape")),
        "print_front": normalize_text(row.get("print_front")),
        "print_back": normalize_text(row.get("print_back")),
        "line_front": normalize_text(row.get("line_front")),
        "line_back": normalize_text(row.get("line_back")),
    }

    normalized["condition_key"] = "|".join(
        [
            normalized["drug_shape"] or "NA",
            normalized["color_class1"] or "NA",
            normalized["color_class2"] or "NA",
            normalized["print_front"] or "NA",
            normalized["print_back"] or "NA",
        ]
    )
    return normalized


def build_summary(rows):
    class_count = Counter()
    color1_count = Counter()
    shape_count = Counter()
    condition_count = Counter()

    for row in rows:
        class_count[row["category_id"]] += 1
        if row["color_class1"]:
            color1_count[row["color_class1"]] += 1
        if row["drug_shape"]:
            shape_count[row["drug_shape"]] += 1
        condition_count[row["condition_key"]] += 1

    rare_conditions = [
        {"condition_key": key, "count": count}
        for key, count in sorted(condition_count.items(), key=lambda item: item[1])
        if count <= 3
    ]

    return {
        "num_rows": len(rows),
        "num_classes": len(class_count),
        "num_primary_colors": len(color1_count),
        "num_shapes": len(shape_count),
        "num_condition_keys": len(condition_count),
        "class_count": dict(sorted(class_count.items())),
        "color_class1_count": dict(sorted(color1_count.items())),
        "drug_shape_count": dict(sorted(shape_count.items())),
        "rare_condition_keys": rare_conditions[:200],
    }


def prepare_pill_generation_dataset(crop_metadata_path, output_dir, verbose=True):
    crop_metadata = load_json(crop_metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_rows = [normalize_metadata_row(row) for row in crop_metadata]
    summary = build_summary(normalized_rows)

    metadata_json_path = output_dir / "metadata.json"
    metadata_csv_path = output_dir / "metadata.csv"
    summary_path = output_dir / "summary.json"

    with open(metadata_json_path, "w", encoding="utf-8") as file:
        json.dump(normalized_rows, file, ensure_ascii=False, indent=2)

    with open(metadata_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(normalized_rows[0].keys()) if normalized_rows else GENERATION_COLUMNS + ["condition_key"])
        writer.writeheader()
        writer.writerows(normalized_rows)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    if verbose:
        print("=" * 55)
        print("pill generation dataset 준비 완료")
        print("=" * 55)
        print(f"  metadata json      : {metadata_json_path}")
        print(f"  metadata csv       : {metadata_csv_path}")
        print(f"  summary            : {summary_path}")
        print(f"  sample 수          : {summary['num_rows']}개")
        print(f"  class 수           : {summary['num_classes']}개")
        print(f"  condition key 수   : {summary['num_condition_keys']}개")

    return {
        "rows": normalized_rows,
        "summary": summary,
        "metadata_json_path": metadata_json_path,
        "metadata_csv_path": metadata_csv_path,
        "summary_path": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare metadata for pill generation modeling.")
    parser.add_argument("--crop-metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    prepare_pill_generation_dataset(
        crop_metadata_path=args.crop_metadata_path,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
