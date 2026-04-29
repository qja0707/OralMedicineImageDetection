"""Build the RT-DETR/Ultralytics dataset for the internet-train experiment.

Train uses manually corrected internet images. Validation uses the original
captured dataset prepared by 01_prepare_original_eval.py.

After the base YOLO dataset is built, this script expands train data with
offline copy-paste augmentation. It crops the manually annotated internet pill
regions, applies strong geometric augmentation only, and pastes them onto
synthetic light backgrounds. No color augmentation is applied.
"""

import json
import random
from pathlib import Path
import sys

import cv2
import numpy as np


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ml.src.data.build_yolo_dataset import build_yolo_dataset


DEFAULT_AUGMENT_FACTOR = 20
DEFAULT_IMAGE_SIZE = 640
DEFAULT_SEED = 42
OBJECT_COUNT_PROBS = ((2, 0.20), (3, 0.30), (4, 0.50))
MAX_OBJECT_IOU = 0.01


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_image_bgr(path):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def build_category_mappings(categories):
    sorted_categories = sorted(categories, key=lambda category: category["id"])
    return {category["id"]: index for index, category in enumerate(sorted_categories)}


def make_background(width, height, rng):
    base = int(rng.integers(205, 236))
    background = np.full((height, width, 3), base, dtype=np.uint8)

    # Add very light measurement-style grid lines without changing object color.
    grid_color = max(170, base - int(rng.integers(18, 34)))
    spacing = int(rng.integers(64, 128))
    offset_x = int(rng.integers(0, spacing))
    offset_y = int(rng.integers(0, spacing))
    for x in range(offset_x, width, spacing):
        cv2.line(background, (x, 0), (x, height), (grid_color, grid_color, grid_color), 1)
    for y in range(offset_y, height, spacing):
        cv2.line(background, (0, y), (width, y), (grid_color, grid_color, grid_color), 1)

    noise = rng.normal(0, 2.0, background.shape).astype(np.int16)
    return np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def crop_with_padding(image, bbox, padding_ratio=0.08):
    x, y, w, h = bbox
    pad = round(max(w, h) * padding_ratio)
    x1 = max(0, round(x) - pad)
    y1 = max(0, round(y) - pad)
    x2 = min(image.shape[1], round(x + w) + pad)
    y2 = min(image.shape[0], round(y + h) + pad)
    crop = image[y1:y2, x1:x2].copy()
    rel_bbox = [round(x - x1), round(y - y1), round(w), round(h)]
    return crop, rel_bbox


def rotate_crop_and_bbox(crop, bbox, angle, border_value):
    height, width = crop.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]

    rotated = cv2.warpAffine(
        crop,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    x, y, w, h = bbox
    corners = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )
    ones = np.ones((4, 1), dtype=np.float32)
    rotated_corners = np.hstack([corners, ones]) @ matrix.T
    x1, y1 = rotated_corners.min(axis=0)
    x2, y2 = rotated_corners.max(axis=0)
    rotated_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
    return rotated, rotated_bbox


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def choose_object_count(rng):
    counts = [item[0] for item in OBJECT_COUNT_PROBS]
    probs = [item[1] for item in OBJECT_COUNT_PROBS]
    return int(rng.choice(counts, p=probs))


def paste_crop(background, crop, bbox, rng, existing_bboxes):
    bg_h, bg_w = background.shape[:2]
    crop_h, crop_w = crop.shape[:2]
    if crop_w >= bg_w or crop_h >= bg_h:
        scale = min((bg_w * 0.8) / crop_w, (bg_h * 0.8) / crop_h)
        new_w = max(1, int(crop_w * scale))
        new_h = max(1, int(crop_h * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        bbox = [value * scale for value in bbox]
        crop_h, crop_w = crop.shape[:2]

    paste_x = None
    paste_y = None
    pasted_bbox = None
    for _ in range(200):
        candidate_x = int(rng.integers(0, bg_w - crop_w + 1))
        candidate_y = int(rng.integers(0, bg_h - crop_h + 1))
        x, y, w, h = bbox
        candidate_bbox = [candidate_x + x, candidate_y + y, w, h]
        if all(bbox_iou(candidate_bbox, existing) <= MAX_OBJECT_IOU for existing in existing_bboxes):
            paste_x = candidate_x
            paste_y = candidate_y
            pasted_bbox = candidate_bbox
            break

    if paste_x is None or paste_y is None:
        return None, None

    background[paste_y : paste_y + crop_h, paste_x : paste_x + crop_w] = crop

    return background, pasted_bbox


def bbox_to_yolo_line(class_id, bbox, image_width, image_height):
    x, y, w, h = bbox
    x = max(0.0, min(float(image_width - 1), x))
    y = max(0.0, min(float(image_height - 1), y))
    w = max(1.0, min(float(image_width) - x, w))
    h = max(1.0, min(float(image_height) - y, h))
    cx = (x + w / 2) / image_width
    cy = (y + h / 2) / image_height
    nw = w / image_width
    nh = h / image_height
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def build_source_objects(coco, images_dir, coco_to_yolo):
    images_by_id = {image["id"]: image for image in coco["images"]}
    objects = []
    for annotation in coco["annotations"]:
        image_info = images_by_id[annotation["image_id"]]
        image = load_image_bgr(images_dir / image_info["file_name"])
        crop, rel_bbox = crop_with_padding(image, annotation["bbox"])
        objects.append(
            {
                "crop": crop,
                "bbox": rel_bbox,
                "class_id": coco_to_yolo[annotation["category_id"]],
                "category_id": annotation["category_id"],
            }
        )
    return objects


def group_objects_by_category(objects):
    grouped = {}
    for obj in objects:
        grouped.setdefault(obj["category_id"], []).append(obj)
    return grouped


def augment_object(source, rng):
    crop = source["crop"]
    bbox = source["bbox"]
    angle = float(rng.uniform(-180.0, 180.0))
    border_value = tuple(int(v) for v in np.median(crop.reshape(-1, 3), axis=0))
    rotated, rotated_bbox = rotate_crop_and_bbox(crop, bbox, angle, border_value)

    scale = float(rng.uniform(0.35, 1.25))
    new_w = max(2, int(rotated.shape[1] * scale))
    new_h = max(2, int(rotated.shape[0] * scale))
    resized = cv2.resize(rotated, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_bbox = [value * scale for value in rotated_bbox]
    return resized, resized_bbox


def add_copy_paste_train_samples(
    coco_path,
    images_dir,
    dataset_dir,
    augment_factor=DEFAULT_AUGMENT_FACTOR,
    image_size=DEFAULT_IMAGE_SIZE,
    seed=DEFAULT_SEED,
):
    coco = load_json(coco_path)
    images_dir = Path(images_dir)
    dataset_dir = Path(dataset_dir)
    train_images_dir = dataset_dir / "images" / "train"
    train_labels_dir = dataset_dir / "labels" / "train"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    coco_to_yolo = build_category_mappings(coco["categories"])
    objects = build_source_objects(coco, images_dir, coco_to_yolo)
    if not objects:
        raise ValueError("No source objects found for copy-paste augmentation")
    objects_by_category = group_objects_by_category(objects)
    category_ids = sorted(objects_by_category)

    target_count = len(coco["images"]) * augment_factor
    generated = 0
    attempts = 0
    max_attempts = target_count * 30

    while generated < target_count and attempts < max_attempts:
        attempts += 1
        background = make_background(image_size, image_size, rng)
        object_count = min(choose_object_count(rng), len(category_ids))
        selected_categories = py_rng.sample(category_ids, object_count)
        label_lines = []
        pasted_bboxes = []
        category_suffix = []

        for category_id in selected_categories:
            source = py_rng.choice(objects_by_category[category_id])
            resized, resized_bbox = augment_object(source, rng)
            background, pasted_bbox = paste_crop(
                background,
                resized,
                resized_bbox,
                rng,
                pasted_bboxes,
            )
            if pasted_bbox is None:
                break

            x, y, w, h = pasted_bbox
            if w < 4 or h < 4 or x < 0 or y < 0 or x + w > image_size or y + h > image_size:
                break

            pasted_bboxes.append(pasted_bbox)
            category_suffix.append(str(category_id))
            label_lines.append(
                bbox_to_yolo_line(source["class_id"], pasted_bbox, image_size, image_size)
            )
        else:
            if len(label_lines) != object_count:
                continue
            generated += 1
            stem = f"copy_paste_{generated:06d}_{'x'.join(category_suffix)}"
            write_image(train_images_dir / f"{stem}.jpg", background)
            (train_labels_dir / f"{stem}.txt").write_text(
                "\n".join(label_lines) + "\n",
                encoding="utf-8",
            )
            continue

    if generated < target_count:
        raise RuntimeError(f"Generated {generated}/{target_count} copy-paste samples")

    metadata = {
        "augment_factor": augment_factor,
        "generated_images": generated,
        "image_size": image_size,
        "seed": seed,
        "object_count_probs": dict(OBJECT_COUNT_PROBS),
        "max_object_iou": MAX_OBJECT_IOU,
        "same_class_duplicate": False,
        "color_augmentation": False,
    }
    with open(dataset_dir / "copy_paste_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("=" * 55)
    print("Copy-paste train 증강 완료")
    print("=" * 55)
    print(f"  원본 train 이미지 수 : {len(coco['images'])}개")
    print(f"  증강 배수            : {augment_factor}배")
    print(f"  생성 이미지 수       : {generated}개")
    print("  이미지당 객체 수     : 2개 20%, 3개 30%, 4개 50%")
    print("  같은 클래스 중복     : 없음")
    print(f"  객체 간 최대 IoU     : {MAX_OBJECT_IOU}")
    print(f"  저장 경로            : {train_images_dir}")


def main():
    data_root = project_root / "ml" / "data"
    dataset_dir = data_root / "processed" / "internet_train_rtdetr"
    train_coco_path = data_root / "interim" / "internet_train" / "internet_train_coco.json"
    train_images_dir = data_root / "raw" / "internet_train" / "images"

    build_yolo_dataset(
        train_coco_path=train_coco_path,
        val_coco_path=data_root / "interim" / "internet_train" / "original_eval_coco.json",
        train_images_dir=train_images_dir,
        val_images_dir=data_root / "raw" / "images",
        output_dir=dataset_dir,
    )
    add_copy_paste_train_samples(
        coco_path=train_coco_path,
        images_dir=train_images_dir,
        dataset_dir=dataset_dir,
    )


if __name__ == "__main__":
    main()
