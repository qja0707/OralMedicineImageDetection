import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_json(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_image(image_path):
    image_path = Path(image_path)
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def write_image(image_path, image):
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError(f"Failed to encode crop: {image_path}")
    encoded.tofile(str(image_path))


def clear_output_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def sanitize_class_dir_name(category_id):
    return str(category_id)


def clamp_bbox(bbox, image_width, image_height):
    x, y, width, height = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image_width, int(round(x + width)))
    y2 = min(image_height, int(round(y + height)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def build_annotation_rows(coco, source_name):
    images_by_id = {image["id"]: image for image in coco.get("images", [])}
    categories_by_id = {
        category["id"]: category.get("name", str(category["id"]))
        for category in coco.get("categories", [])
    }

    rows = []
    for annotation in coco.get("annotations", []):
        image = images_by_id.get(annotation["image_id"])
        if image is None:
            continue

        category_id = annotation["category_id"]
        rows.append(
            {
                "annotation_id": annotation["id"],
                "source_name": source_name,
                "image_id": image["id"],
                "file_name": image["file_name"],
                "image_width": image["width"],
                "image_height": image["height"],
                "category_id": category_id,
                "category_name": categories_by_id.get(category_id, str(category_id)),
                "bbox": annotation["bbox"],
            }
        )

    return rows


def split_rows_by_class(rows, val_ratio, seed):
    rng = random.Random(seed)
    rows_by_category = defaultdict(list)
    for row in rows:
        rows_by_category[row["category_id"]].append(row)

    train_rows = []
    val_rows = []

    for category_id in sorted(rows_by_category):
        category_rows = list(rows_by_category[category_id])
        rng.shuffle(category_rows)

        if len(category_rows) < 2 or val_ratio <= 0:
            train_rows.extend(category_rows)
            continue

        val_count = max(1, round(len(category_rows) * val_ratio))
        val_count = min(val_count, len(category_rows) - 1)

        val_rows.extend(category_rows[:val_count])
        train_rows.extend(category_rows[val_count:])

    return train_rows, val_rows


def crop_rows(rows, output_dir, split_name):
    output_dir = Path(output_dir)
    image_cache = {}
    converted = 0
    skipped = 0

    for row in rows:
        source_image_path = Path(row["raw_images_dir"]) / row["file_name"]
        if source_image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped += 1
            continue
        if not source_image_path.exists():
            skipped += 1
            continue

        if source_image_path not in image_cache:
            try:
                image_cache[source_image_path] = read_image(source_image_path)
            except ValueError:
                skipped += 1
                continue

        image = image_cache[source_image_path]
        image_height, image_width = image.shape[:2]
        clamped_bbox = clamp_bbox(row["bbox"], image_width, image_height)
        if clamped_bbox is None:
            skipped += 1
            continue

        x1, y1, x2, y2 = clamped_bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            skipped += 1
            continue

        class_dir = sanitize_class_dir_name(row["category_id"])
        source_stem = Path(row["file_name"]).stem
        crop_name = f"{row['source_name']}_{source_stem}_ann{row['annotation_id']:06d}.jpg"
        target_path = output_dir / split_name / class_dir / crop_name
        write_image(target_path, crop)
        converted += 1

    return {"converted": converted, "skipped": skipped}


def write_class_mapping(output_dir, categories):
    output_dir = Path(output_dir)
    categories_by_id = {}
    for category in categories:
        categories_by_id.setdefault(category["id"], category)

    sorted_categories = sorted(categories_by_id.values(), key=lambda category: category["id"])
    mapping = {
        str(category["id"]): category.get("name", str(category["id"]))
        for category in sorted_categories
    }
    mapping_path = output_dir / "class_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return mapping_path


def normalize_source(source):
    if isinstance(source, dict):
        name = source["name"]
        coco_path = source["coco_path"]
        raw_images_dir = source["raw_images_dir"]
    else:
        name, coco_path, raw_images_dir = source

    return {
        "name": str(name),
        "coco_path": Path(coco_path),
        "raw_images_dir": Path(raw_images_dir),
    }


def load_rows_from_sources(sources, skip_missing=False):
    all_rows = []
    all_categories = []
    loaded_sources = []
    skipped_sources = []

    for source in sources:
        source = normalize_source(source)
        coco_path = source["coco_path"]
        raw_images_dir = source["raw_images_dir"]

        if not coco_path.exists() or not raw_images_dir.exists():
            if skip_missing:
                skipped_sources.append(source)
                continue
            if not coco_path.exists():
                raise FileNotFoundError(f"COCO file not found: {coco_path}")
            raise FileNotFoundError(f"Raw images directory not found: {raw_images_dir}")

        coco = load_json(coco_path)
        source_rows = build_annotation_rows(coco, source_name=source["name"])
        for row in source_rows:
            row["raw_images_dir"] = raw_images_dir

        all_rows.extend(source_rows)
        all_categories.extend(coco.get("categories", []))
        loaded_sources.append(
            {
                **source,
                "images": len(coco.get("images", [])),
                "annotations": len(source_rows),
                "categories": len(coco.get("categories", [])),
            }
        )

    return all_rows, all_categories, loaded_sources, skipped_sources


def build_yolo_classification_dataset_from_sources(
    sources,
    output_dir,
    val_ratio=0.2,
    seed=42,
    skip_missing=False,
    verbose=True,
):
    output_dir = Path(output_dir)
    rows, categories, loaded_sources, skipped_sources = load_rows_from_sources(
        sources=sources,
        skip_missing=skip_missing,
    )
    if not loaded_sources:
        raise FileNotFoundError("No dataset sources were loaded.")

    train_rows, val_rows = split_rows_by_class(rows, val_ratio=val_ratio, seed=seed)

    clear_output_dir(output_dir)
    train_stats = crop_rows(train_rows, output_dir, "train")
    val_stats = crop_rows(val_rows, output_dir, "val")
    mapping_path = write_class_mapping(output_dir, categories)

    if verbose:
        print("=" * 60)
        print("COCO bbox -> YOLO classification crop dataset")
        print("=" * 60)
        print(f"Output dir       : {output_dir}")
        print("Loaded sources   :")
        for source in loaded_sources:
            print(
                f"  - {source['name']}: images={source['images']}, "
                f"annotations={source['annotations']}, categories={source['categories']}"
            )
        if skipped_sources:
            print("Skipped sources  :")
            for source in skipped_sources:
                print(f"  - {source['name']}: {source['coco_path']} / {source['raw_images_dir']}")
        print(f"Classes          : {len({category['id'] for category in categories})}")
        print(f"Annotations      : {len(rows)}")
        print(f"Train converted  : {train_stats['converted']}")
        print(f"Train skipped    : {train_stats['skipped']}")
        print(f"Val converted    : {val_stats['converted']}")
        print(f"Val skipped      : {val_stats['skipped']}")
        print(f"Class mapping    : {mapping_path}")

    return {
        "train_stats": train_stats,
        "val_stats": val_stats,
        "mapping_path": mapping_path,
        "output_dir": output_dir,
    }


def build_yolo_classification_dataset(
    coco_path,
    raw_images_dir,
    output_dir,
    val_ratio=0.2,
    seed=42,
    verbose=True,
):
    return build_yolo_classification_dataset_from_sources(
        sources=[
            {
                "name": Path(coco_path).stem,
                "coco_path": coco_path,
                "raw_images_dir": raw_images_dir,
            }
        ],
        output_dir=output_dir,
        val_ratio=val_ratio,
        seed=seed,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build a YOLO classification dataset by cropping COCO bboxes without padding."
    )
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        metavar=("NAME", "COCO_PATH", "RAW_IMAGES_DIR"),
        help="Dataset source. Can be passed multiple times.",
    )
    parser.add_argument("--coco-path", help="Single COCO path. Kept for backward compatibility.")
    parser.add_argument("--raw-images-dir", help="Single raw image directory. Kept for backward compatibility.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.source:
        sources = [
            {"name": name, "coco_path": coco_path, "raw_images_dir": raw_images_dir}
            for name, coco_path, raw_images_dir in args.source
        ]
    elif args.coco_path and args.raw_images_dir:
        sources = [
            {
                "name": Path(args.coco_path).stem,
                "coco_path": args.coco_path,
                "raw_images_dir": args.raw_images_dir,
            }
        ]
    else:
        parser.error("Provide either --source or both --coco-path and --raw-images-dir.")

    build_yolo_classification_dataset_from_sources(
        sources=sources,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        skip_missing=args.skip_missing,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
