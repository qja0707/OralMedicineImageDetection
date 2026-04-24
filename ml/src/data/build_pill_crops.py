import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def load_json(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def merge_metadata_by_annotation(coco, annotation_metadata):
    metadata_by_annotation_id = {
        item["annotation_id"]: item for item in annotation_metadata
    }
    image_by_id = {image["id"]: image for image in coco["images"]}

    merged_rows = []
    for annotation in coco["annotations"]:
        annotation_id = annotation["id"]
        metadata = dict(metadata_by_annotation_id.get(annotation_id, {}))
        image_info = image_by_id.get(annotation["image_id"])

        if image_info is None:
            continue

        metadata.setdefault("annotation_id", annotation_id)
        metadata.setdefault("image_id", annotation["image_id"])
        metadata.setdefault("category_id", annotation["category_id"])
        metadata.setdefault("file_name", image_info["file_name"])
        metadata["bbox"] = annotation["bbox"]
        metadata["area"] = annotation.get("area")
        metadata["iscrowd"] = annotation.get("iscrowd", 0)
        metadata["image_width"] = image_info["width"]
        metadata["image_height"] = image_info["height"]
        merged_rows.append(metadata)

    return merged_rows


def crop_bbox(image, bbox, margin_ratio):
    image_width, image_height = image.size
    x, y, w, h = bbox
    margin_x = w * margin_ratio
    margin_y = h * margin_ratio

    left = max(0, int(x - margin_x))
    top = max(0, int(y - margin_y))
    right = min(image_width, int(x + w + margin_x))
    bottom = min(image_height, int(y + h + margin_y))

    if right <= left or bottom <= top:
        return None, None

    crop = image.crop((left, top, right, bottom))
    crop_box = {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
    }
    return crop, crop_box


def build_pill_crops(
    coco_json_path,
    annotation_metadata_path,
    raw_images_dir,
    output_dir,
    margin_ratio=0.1,
    verbose=True,
):
    coco = load_json(coco_json_path)
    annotation_metadata = load_json(annotation_metadata_path)

    raw_images_dir = Path(raw_images_dir)
    output_dir = Path(output_dir)
    crops_dir = output_dir / "images"
    crops_dir.mkdir(parents=True, exist_ok=True)

    merged_rows = merge_metadata_by_annotation(coco, annotation_metadata)

    def log(message):
        if verbose:
            print(message)

    log("=" * 55)
    log("1단계: annotation 단위 crop dataset 생성 중...")
    log("=" * 55)
    log(f"  annotation 수      : {len(merged_rows)}개")
    log(f"  margin ratio       : {margin_ratio}")

    crop_metadata_rows = []
    skipped = 0

    for row in merged_rows:
        source_image_path = raw_images_dir / row["file_name"]
        if not source_image_path.exists():
            skipped += 1
            continue

        image = Image.open(source_image_path).convert("RGB")
        crop_image, crop_box = crop_bbox(image, row["bbox"], margin_ratio)
        if crop_image is None:
            skipped += 1
            continue

        crop_file_name = f"ann_{row['annotation_id']}_cat_{row['category_id']}.png"
        crop_path = crops_dir / crop_file_name
        crop_image.save(crop_path)

        crop_row = dict(row)
        crop_row["crop_file_name"] = crop_file_name
        crop_row["crop_relative_path"] = f"images/{crop_file_name}"
        crop_row["crop_left"] = crop_box["left"]
        crop_row["crop_top"] = crop_box["top"]
        crop_row["crop_right"] = crop_box["right"]
        crop_row["crop_bottom"] = crop_box["bottom"]
        crop_row["crop_width"] = crop_box["width"]
        crop_row["crop_height"] = crop_box["height"]
        crop_metadata_rows.append(crop_row)

    json_path = output_dir / "crop_metadata.json"
    csv_path = output_dir / "crop_metadata.csv"
    summary_path = output_dir / "crop_summary.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(crop_metadata_rows, file, ensure_ascii=False, indent=2)

    if crop_metadata_rows:
        fieldnames = sorted(crop_metadata_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(crop_metadata_rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    summary = {
        "num_annotations": len(merged_rows),
        "num_crops": len(crop_metadata_rows),
        "num_skipped": skipped,
        "margin_ratio": margin_ratio,
        "output_dir": str(output_dir),
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    log("\n" + "=" * 55)
    log("2단계: crop dataset 저장 완료")
    log("=" * 55)
    log(f"  crop image dir     : {crops_dir}")
    log(f"  crop metadata json : {json_path}")
    log(f"  crop metadata csv  : {csv_path}")
    log(f"  summary            : {summary_path}")
    log(f"  생성된 crop 수     : {len(crop_metadata_rows)}개")
    log(f"  스킵 수            : {skipped}개")

    return {
        "rows": crop_metadata_rows,
        "json_path": json_path,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Build annotation-level pill crop dataset.")
    parser.add_argument("--coco-json-path", required=True)
    parser.add_argument("--annotation-metadata-path", required=True)
    parser.add_argument("--raw-images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin-ratio", type=float, default=0.1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    build_pill_crops(
        coco_json_path=args.coco_json_path,
        annotation_metadata_path=args.annotation_metadata_path,
        raw_images_dir=args.raw_images_dir,
        output_dir=args.output_dir,
        margin_ratio=args.margin_ratio,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
