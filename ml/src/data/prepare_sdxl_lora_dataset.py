import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_PROMPT_TEMPLATE = "a close-up product photo of {color} {shape} pill"


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


def slugify_text(value, fallback="na"):
    normalized = normalize_text(value).lower()
    if not normalized:
        return fallback

    chars = []
    prev_sep = False
    for char in normalized:
        if char.isascii() and char.isalnum():
            chars.append(char)
            prev_sep = False
            continue

        if prev_sep:
            continue

        chars.append("-")
        prev_sep = True

    slug = "".join(chars).strip("-")
    return slug or fallback


def build_condition_key(row):
    return "|".join(
        [
            row["drug_shape"] or "NA",
            row["color_class1"] or "NA",
            row["color_class2"] or "NA",
            row["print_front"] or "NA",
            row["print_back"] or "NA",
        ]
    )


def normalize_generation_row(row):
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
    normalized["condition_key"] = build_condition_key(normalized)
    return normalized


def normalize_crop_row(row):
    normalized = normalize_generation_row(row)
    normalized["file_name"] = normalize_text(row.get("file_name"))
    normalized["image_id"] = row.get("image_id")
    return normalized


def load_rows(metadata_path):
    rows = load_json(metadata_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON: {metadata_path}")
    if not rows:
        return []

    first_row = rows[0]
    if "crop_relative_path" not in first_row:
        raise ValueError(
            "Metadata rows must include 'crop_relative_path'. Run build_pill_crops first."
        )

    if "condition_key" in first_row:
        return [normalize_generation_row(row) for row in rows]

    return [normalize_crop_row(row) for row in rows]


def infer_crop_root(metadata_path):
    metadata_path = Path(metadata_path)
    parent = metadata_path.parent

    if parent.name == "pill_crops":
        return parent
    if parent.name == "pill_generation":
        sibling = parent.parent / "pill_crops"
        if sibling.exists():
            return sibling

    if (parent / "images").exists():
        return parent

    sibling = parent.parent / "pill_crops"
    return sibling


def build_caption(row, prompt_template):
    color = row["color_class1"] or "plain"
    shape = row["drug_shape"] or "unknown-shape"

    caption = prompt_template.format(
        color=color,
        shape=shape,
        color2=row["color_class2"] or "none",
        print_front=row["print_front"] or "none",
        print_back=row["print_back"] or "none",
        line_front=row["line_front"] or "none",
        line_back=row["line_back"] or "none",
        category_id=row["category_id"],
        condition_key=row["condition_key"],
    )
    return normalize_text(caption)


def build_tags(row):
    tags = [
        f"shape:{slugify_text(row['drug_shape'])}",
        f"color:{slugify_text(row['color_class1'])}",
    ]

    if row["color_class2"]:
        tags.append(f"color2:{slugify_text(row['color_class2'])}")
    if row["print_front"]:
        tags.append(f"print-front:{slugify_text(row['print_front'])}")
    if row["print_back"]:
        tags.append(f"print-back:{slugify_text(row['print_back'])}")
    if row["line_front"]:
        tags.append(f"line-front:{slugify_text(row['line_front'])}")
    if row["line_back"]:
        tags.append(f"line-back:{slugify_text(row['line_back'])}")

    tags.append(f"class:{row['category_id']}")
    return ", ".join(tags)


def prepare_sdxl_lora_dataset(
    metadata_path,
    crop_root_dir,
    output_dir,
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    include_imprint_tags=False,
    verbose=True,
):
    rows = load_rows(metadata_path)
    crop_root_dir = Path(crop_root_dir)
    output_dir = Path(output_dir)

    if not crop_root_dir.exists():
        raise FileNotFoundError(f"Crop root dir not found: {crop_root_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    captions_dir = output_dir / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    metadata_jsonl_path = output_dir / "metadata.jsonl"
    manifest_csv_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"

    class_count = Counter()
    condition_count = Counter()
    missing_images = []
    manifest_rows = []

    with open(metadata_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for row in rows:
            source_image_path = crop_root_dir / row["crop_relative_path"]
            if not source_image_path.exists():
                missing_images.append(
                    {
                        "annotation_id": row["annotation_id"],
                        "category_id": row["category_id"],
                        "crop_relative_path": row["crop_relative_path"],
                    }
                )
                continue

            image_name = (
                f"ann_{row['annotation_id']}_cat_{row['category_id']}{source_image_path.suffix}"
            )
            output_image_path = images_dir / image_name
            shutil.copy2(source_image_path, output_image_path)

            caption = build_caption(row, prompt_template)
            if include_imprint_tags:
                tags = build_tags(row)
                caption = f"{caption}, {tags}"

            caption_path = captions_dir / f"{Path(image_name).stem}.txt"
            caption_path.write_text(caption + "\n", encoding="utf-8")

            relative_image_path = Path("images") / image_name
            record = {
                "file_name": relative_image_path.as_posix(),
                "text": caption,
                "annotation_id": row["annotation_id"],
                "category_id": row["category_id"],
                "condition_key": row["condition_key"],
                "color_class1": row["color_class1"],
                "color_class2": row["color_class2"],
                "drug_shape": row["drug_shape"],
                "print_front": row["print_front"],
                "print_back": row["print_back"],
                "line_front": row["line_front"],
                "line_back": row["line_back"],
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            manifest_rows.append(
                {
                    "annotation_id": row["annotation_id"],
                    "category_id": row["category_id"],
                    "image_path": relative_image_path.as_posix(),
                    "caption_path": (Path("captions") / f"{Path(image_name).stem}.txt").as_posix(),
                    "caption": caption,
                    "condition_key": row["condition_key"],
                    "color_class1": row["color_class1"],
                    "color_class2": row["color_class2"],
                    "drug_shape": row["drug_shape"],
                    "print_front": row["print_front"],
                    "print_back": row["print_back"],
                    "line_front": row["line_front"],
                    "line_back": row["line_back"],
                }
            )

            class_count[row["category_id"]] += 1
            condition_count[row["condition_key"]] += 1

    fieldnames = [
        "annotation_id",
        "category_id",
        "image_path",
        "caption_path",
        "caption",
        "condition_key",
        "color_class1",
        "color_class2",
        "drug_shape",
        "print_front",
        "print_back",
        "line_front",
        "line_back",
    ]
    with open(manifest_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "num_input_rows": len(rows),
        "num_output_rows": len(manifest_rows),
        "num_missing_images": len(missing_images),
        "num_classes": len(class_count),
        "num_condition_keys": len(condition_count),
        "prompt_template": prompt_template,
        "include_imprint_tags": include_imprint_tags,
        "crop_root_dir": str(crop_root_dir),
        "output_dir": str(output_dir),
        "missing_images": missing_images[:200],
        "class_count": dict(sorted(class_count.items())),
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    if verbose:
        print("=" * 55)
        print("SDXL LoRA dataset 준비 완료")
        print("=" * 55)
        print(f"  input rows         : {len(rows)}")
        print(f"  output rows        : {len(manifest_rows)}")
        print(f"  missing images     : {len(missing_images)}")
        print(f"  images dir         : {images_dir}")
        print(f"  captions dir       : {captions_dir}")
        print(f"  metadata jsonl     : {metadata_jsonl_path}")
        print(f"  manifest csv       : {manifest_csv_path}")
        print(f"  summary            : {summary_path}")

    return {
        "rows": manifest_rows,
        "summary": summary,
        "metadata_jsonl_path": metadata_jsonl_path,
        "manifest_csv_path": manifest_csv_path,
        "summary_path": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare SDXL LoRA dataset from pill crop metadata.")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--crop-root-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--include-imprint-tags", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    crop_root_dir = args.crop_root_dir or infer_crop_root(args.metadata_path)
    prepare_sdxl_lora_dataset(
        metadata_path=args.metadata_path,
        crop_root_dir=crop_root_dir,
        output_dir=args.output_dir,
        prompt_template=args.prompt_template,
        include_imprint_tags=args.include_imprint_tags,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
