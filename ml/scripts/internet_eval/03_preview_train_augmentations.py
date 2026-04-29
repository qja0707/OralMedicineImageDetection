import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"COCO json not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_image_rgb(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_image_rgb(path, image_rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image_bgr)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def build_transform(image_size):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=1.0,
            ),
            A.Affine(
                scale=(0.7, 1.3),
                translate_percent=(-0.1, 0.1),
                rotate=(-180, 180),
                fit_output=False,
                p=0.9,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=18,
                sat_shift_limit=70,
                val_shift_limit=40,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.15,
        ),
    )


def draw_bboxes(image_rgb, bboxes, category_ids):
    preview = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for index, (bbox, category_id) in enumerate(zip(bboxes, category_ids), start=1):
        x, y, w, h = [round(value) for value in bbox]
        x2 = x + w
        y2 = y + h
        cv2.rectangle(preview, (x, y), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            preview,
            f"{category_id}:{index}",
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def group_annotations(coco):
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    return annotations_by_image


def preview_augmentations(
    coco_path,
    images_dir,
    output_dir,
    num_samples,
    image_size,
    seed,
):
    coco = load_json(coco_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    annotations_by_image = group_annotations(coco)
    candidate_images = [
        image for image in coco["images"] if annotations_by_image.get(image["id"])
    ]
    if not candidate_images:
        raise ValueError("No annotated images found in COCO json")

    transform = build_transform(image_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for sample_index in range(1, num_samples + 1):
        image_info = random.choice(candidate_images)
        annotations = annotations_by_image[image_info["id"]]
        image_path = images_dir / image_info["file_name"]
        image_rgb = load_image_rgb(image_path)

        bboxes = [annotation["bbox"] for annotation in annotations]
        category_ids = [annotation["category_id"] for annotation in annotations]
        augmented = transform(
            image=image_rgb,
            bboxes=bboxes,
            category_ids=category_ids,
        )

        preview = draw_bboxes(
            augmented["image"],
            augmented["bboxes"],
            augmented["category_ids"],
        )
        stem = Path(image_info["file_name"]).stem
        output_path = output_dir / f"{sample_index:03d}_{stem}_aug.jpg"
        write_image_rgb(output_path, preview)
        saved.append(output_path)

    return saved


def main():
    data_root = project_root / "ml" / "data"
    parser = argparse.ArgumentParser(
        description="Save augmented bbox preview samples for internet train COCO."
    )
    parser.add_argument(
        "--coco-path",
        type=Path,
        default=data_root / "interim" / "internet_eval" / "internet_train_coco.json",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=data_root / "raw" / "internet_eval" / "images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "interim" / "internet_eval" / "augmentation_preview",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    saved = preview_augmentations(
        coco_path=args.coco_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        seed=args.seed,
    )

    print("=" * 55)
    print("증강 bbox 프리뷰 저장 완료")
    print("=" * 55)
    print(f"  COCO      : {args.coco_path}")
    print(f"  Images    : {args.images_dir}")
    print(f"  Output    : {args.output_dir}")
    print(f"  Samples   : {len(saved)}")


if __name__ == "__main__":
    main()
