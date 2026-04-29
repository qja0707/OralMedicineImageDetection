import csv
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_internet_mapping(mapping_csv_path):
    mapping_csv_path = Path(mapping_csv_path)
    if not mapping_csv_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv_path}")

    rows = []
    with open(mapping_csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_fields = {"image_file", "category_id", "category_name"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"Mapping CSV missing fields: {sorted(missing_fields)}")

        for row in reader:
            image_file = row["image_file"].strip()
            category_id = int(row["category_id"])
            category_name = row["category_name"].strip()
            rows.append(
                {
                    "image_file": image_file,
                    "category_id": category_id,
                    "category_name": category_name,
                }
            )

    return rows


def load_image_rgb(image_path):
    image_path = Path(image_path)
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def estimate_border_background(image_rgb, border_ratio=0.06):
    height, width = image_rgb.shape[:2]
    border = max(8, round(min(width, height) * border_ratio))
    border_pixels = np.concatenate(
        [
            image_rgb[:border, :, :].reshape(-1, 3),
            image_rgb[-border:, :, :].reshape(-1, 3),
            image_rgb[:, :border, :].reshape(-1, 3),
            image_rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border_pixels, axis=0)


def build_foreground_mask(image_rgb, open_scale=0.035, close_scale=0.025):
    background = estimate_border_background(image_rgb)
    diff = np.linalg.norm(image_rgb.astype(np.float32) - background, axis=2)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]

    threshold = max(18.0, float(np.percentile(diff, 70)))
    mask = ((diff > threshold) & (value < 252)).astype(np.uint8) * 255

    height, width = mask.shape
    border = max(4, round(min(width, height) * 0.015))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    kernel_size = max(5, round(min(width, height) * open_scale))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    close_kernel_size = max(5, round(min(width, height) * close_scale))
    if close_kernel_size % 2 == 0:
        close_kernel_size += 1
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def merge_overlapping_candidates(candidates):
    merged = []

    for candidate in sorted(candidates, key=lambda item: item["area"], reverse=True):
        x, y, w, h = candidate["bbox"]
        candidate_box = np.array([x, y, x + w, y + h], dtype=np.float32)
        merged_into_existing = False

        for existing in merged:
            ex, ey, ew, eh = existing["bbox"]
            existing_box = np.array([ex, ey, ex + ew, ey + eh], dtype=np.float32)

            ix1 = max(candidate_box[0], existing_box[0])
            iy1 = max(candidate_box[1], existing_box[1])
            ix2 = min(candidate_box[2], existing_box[2])
            iy2 = min(candidate_box[3], existing_box[3])
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            smaller_area = min(w * h, ew * eh)

            if smaller_area > 0 and intersection / smaller_area > 0.15:
                nx1 = min(candidate_box[0], existing_box[0])
                ny1 = min(candidate_box[1], existing_box[1])
                nx2 = max(candidate_box[2], existing_box[2])
                ny2 = max(candidate_box[3], existing_box[3])
                existing["bbox"] = [float(nx1), float(ny1), float(nx2 - nx1), float(ny2 - ny1)]
                existing["area"] += candidate["area"]
                existing["box_area"] = float((nx2 - nx1) * (ny2 - ny1))
                merged_into_existing = True
                break

        if not merged_into_existing:
            merged.append(dict(candidate))

    return merged


def find_bbox_candidates(image_rgb, min_area_ratio, open_scale, close_scale):
    height, width = image_rgb.shape[:2]
    image_area = width * height
    min_area = image_area * min_area_ratio
    mask = build_foreground_mask(
        image_rgb,
        open_scale=open_scale,
        close_scale=close_scale,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        box_area = w * h
        if box_area > image_area * 0.55:
            continue

        aspect_ratio = w / h
        if aspect_ratio < 0.35 or aspect_ratio > 6.0:
            continue

        candidates.append(
            {
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(area),
                "box_area": float(box_area),
            }
        )

    if not candidates:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            candidates.append(
                {
                    "bbox": [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)],
                    "area": float(len(xs)),
                    "box_area": float((x2 - x1 + 1) * (y2 - y1 + 1)),
                }
            )

    candidates = merge_overlapping_candidates(candidates)
    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates, mask


def detect_pill_bboxes(image_rgb, min_area_ratio=0.003, max_bboxes=2):
    candidates, mask = find_bbox_candidates(
        image_rgb,
        min_area_ratio=min_area_ratio,
        open_scale=0.035,
        close_scale=0.025,
    )

    if not candidates:
        candidates, mask = find_bbox_candidates(
            image_rgb,
            min_area_ratio=0.0015,
            open_scale=0.018,
            close_scale=0.012,
        )

    return candidates[:max_bboxes], mask


def draw_preview(image_rgb, bboxes, category_id, output_path):
    preview = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for index, bbox in enumerate(bboxes, start=1):
        x, y, w, h = [round(value) for value in bbox]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(
            preview,
            f"{category_id}:{index}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, preview)
    if not ok:
        raise ValueError(f"Failed to encode preview: {output_path}")
    encoded.tofile(str(output_path))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_internet_train_coco(
    internet_images_dir,
    mapping_csv_path,
    output_dir,
    output_filename="internet_train_coco.json",
    metadata_filename="internet_train_bbox_metadata.json",
    preview_dirname="bbox_preview",
    verbose=True,
):
    internet_images_dir = Path(internet_images_dir)
    output_dir = Path(output_dir)
    preview_dir = output_dir / preview_dirname
    coco_output_path = output_dir / output_filename
    metadata_output_path = output_dir / metadata_filename

    if not internet_images_dir.exists():
        raise FileNotFoundError(f"Internet images directory not found: {internet_images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    def log(message):
        if verbose:
            print(message)

    rows = load_internet_mapping(mapping_csv_path)
    categories = [
        {"id": row["category_id"], "name": row["category_name"]}
        for row in sorted(rows, key=lambda item: item["category_id"])
    ]

    images = []
    annotations = []
    bbox_metadata = []
    annotation_id = 1

    log("=" * 55)
    log("인터넷 원본 이미지 자동 bbox 생성 중...")
    log("=" * 55)
    log(f"  입력 이미지 폴더: {internet_images_dir}")
    log(f"  매핑 CSV        : {mapping_csv_path}")
    log(f"  대상 클래스 수  : {len(rows)}개")

    for image_id, row in enumerate(rows, start=1):
        image_file = row["image_file"]
        category_id = row["category_id"]
        source_image_path = internet_images_dir / image_file

        if source_image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source_image_path}")
        if not source_image_path.exists():
            raise FileNotFoundError(f"Image not found: {source_image_path}")

        image_rgb = load_image_rgb(source_image_path)
        height, width = image_rgb.shape[:2]
        candidates, _ = detect_pill_bboxes(image_rgb)
        if not candidates:
            raise ValueError(f"No bbox candidates found: {source_image_path}")

        images.append(
            {
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height,
            }
        )

        image_bboxes = []
        for candidate in candidates:
            x, y, w, h = candidate["bbox"]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            image_bboxes.append([x, y, w, h])
            annotation_id += 1

        preview_path = preview_dir / f"{Path(image_file).stem}_bbox.jpg"
        draw_preview(image_rgb, image_bboxes, category_id, preview_path)

        bbox_metadata.append(
            {
                "image_file": image_file,
                "category_id": category_id,
                "category_name": row["category_name"],
                "width": width,
                "height": height,
                "num_bboxes": len(image_bboxes),
                "bboxes": image_bboxes,
                "preview_file": str(preview_path),
            }
        )

        log(f"  [{category_id}] {image_file}: bbox {len(image_bboxes)}개")

    coco = {
        "info": {
            "description": "Internet pill images with automatically generated bboxes",
            "bbox_method": "border-background color difference connected components",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    write_json(coco_output_path, coco)
    write_json(metadata_output_path, bbox_metadata)

    log("\n" + "=" * 55)
    log("인터넷 train COCO 저장 완료")
    log("=" * 55)
    log(f"  COCO JSON   : {coco_output_path}")
    log(f"  Metadata    : {metadata_output_path}")
    log(f"  Preview dir : {preview_dir}")
    log(f"  이미지 수   : {len(images)}개")
    log(f"  bbox 수     : {len(annotations)}개")

    return {
        "coco_output_path": coco_output_path,
        "metadata_output_path": metadata_output_path,
        "preview_dir": preview_dir,
        "num_images": len(images),
        "num_annotations": len(annotations),
    }
