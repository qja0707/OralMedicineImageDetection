import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def read_image_rgb(image_path):
    image_path = Path(image_path)
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_image_rgb(image_path, image_rgb):
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError(f"Failed to encode image: {image_path}")
    encoded.tofile(str(image_path))


def list_images(split_dir):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(
        path
        for path in split_dir.glob("*/*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_transform(enable_horizontal_flip=False, enable_vertical_flip=False):
    transforms = [
        A.Affine(
            scale=(0.9, 1.12),
            translate_percent=(-0.04, 0.04),
            rotate=(-180, 180),
            shear=(-3, 3),
            fit_output=False,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.9,
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                A.ImageCompression(quality_range=(78, 96), p=1.0),
            ],
            p=0.35,
        ),
    ]

    if enable_horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if enable_vertical_flip:
        transforms.append(A.VerticalFlip(p=0.5))

    return A.Compose(transforms)


def clear_output_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_split(input_split_dir, output_split_dir):
    copied = 0
    for image_path in list_images(input_split_dir):
        class_name = image_path.parent.name
        target_path = Path(output_split_dir) / class_name / image_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_path)
        copied += 1
    return copied


def copy_metadata(input_dir, output_dir):
    for metadata_name in ["class_mapping.json"]:
        source = Path(input_dir) / metadata_name
        if source.exists():
            shutil.copy2(source, Path(output_dir) / metadata_name)


def group_images_by_class(image_paths):
    images_by_class = defaultdict(list)
    for image_path in image_paths:
        images_by_class[image_path.parent.name].append(image_path)
    return {class_name: sorted(paths) for class_name, paths in images_by_class.items()}


def infer_source_name(image_path, source_names):
    stem = Path(image_path).stem
    for source_name in source_names:
        if stem == source_name or stem.startswith(f"{source_name}_"):
            return source_name
    return stem.split("_", 1)[0] if "_" in stem else "unknown"


def group_images_by_source(image_paths, source_names):
    images_by_source = defaultdict(list)
    for image_path in image_paths:
        source_name = infer_source_name(image_path, source_names)
        images_by_source[source_name].append(image_path)
    return {source_name: sorted(paths) for source_name, paths in images_by_source.items()}


def write_balance_report(output_dir, class_stats):
    report_path = Path(output_dir) / "balance_report.json"
    report_path.write_text(
        json.dumps(class_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def augment_train_split(
    input_dir,
    output_dir,
    copies_per_image=5,
    seed=42,
    keep_original=True,
    enable_horizontal_flip=False,
    enable_vertical_flip=False,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    train_dir = input_dir / "train"
    val_dir = input_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train split not found: {train_dir}")

    random.seed(seed)
    np.random.seed(seed)
    transform = build_transform(
        enable_horizontal_flip=enable_horizontal_flip,
        enable_vertical_flip=enable_vertical_flip,
    )

    clear_output_dir(output_dir)
    copy_metadata(input_dir, output_dir)

    copied_train = copy_split(train_dir, output_dir / "train") if keep_original else 0
    copied_val = copy_split(val_dir, output_dir / "val")

    train_images = list_images(train_dir)
    augmented = 0
    skipped = 0

    for image_path in train_images:
        class_name = image_path.parent.name
        try:
            image_rgb = read_image_rgb(image_path)
        except ValueError:
            skipped += 1
            continue

        for copy_index in range(1, copies_per_image + 1):
            augmented_image = transform(image=image_rgb)["image"]
            target_name = f"{image_path.stem}_aug{copy_index:02d}.jpg"
            target_path = output_dir / "train" / class_name / target_name
            write_image_rgb(target_path, augmented_image)
            augmented += 1

    return {
        "input_train": len(train_images),
        "copied_train": copied_train,
        "copied_val": copied_val,
        "augmented": augmented,
        "skipped": skipped,
        "output_dir": output_dir,
    }


def balance_train_split(
    input_dir,
    output_dir,
    target_count=None,
    seed=42,
    keep_original=True,
    enable_horizontal_flip=False,
    enable_vertical_flip=False,
    balance_sources=False,
    source_names=("kaggle", "internet"),
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    train_dir = input_dir / "train"
    val_dir = input_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train split not found: {train_dir}")

    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    transform = build_transform(
        enable_horizontal_flip=enable_horizontal_flip,
        enable_vertical_flip=enable_vertical_flip,
    )

    clear_output_dir(output_dir)
    copy_metadata(input_dir, output_dir)

    copied_train = copy_split(train_dir, output_dir / "train") if keep_original else 0
    copied_val = copy_split(val_dir, output_dir / "val")

    train_images = list_images(train_dir)
    images_by_class = group_images_by_class(train_images)
    if not images_by_class:
        raise ValueError(f"No train images found: {train_dir}")

    resolved_target_count = target_count or max(len(paths) for paths in images_by_class.values())
    if resolved_target_count < 1:
        raise ValueError("target_count must be at least 1")

    augmented = 0
    skipped = 0
    class_stats = {}

    source_names = tuple(source_names)

    for class_name in sorted(images_by_class):
        class_images = images_by_class[class_name]
        original_count = len(class_images)
        generated = 0
        source_stats = {}

        if balance_sources:
            images_by_source = group_images_by_source(class_images, source_names)
            present_sources = {
                source_name: paths
                for source_name, paths in images_by_source.items()
                if paths
            }
            per_source_target = max(
                max(len(paths) for paths in present_sources.values()),
                int(np.ceil(resolved_target_count / max(1, len(present_sources)))),
            )

            for source_name in sorted(present_sources):
                source_images = present_sources[source_name]
                source_original_count = len(source_images)
                source_needed = max(0, per_source_target - source_original_count)
                source_generated = 0

                for index in range(1, source_needed + 1):
                    image_path = rng.choice(source_images)
                    try:
                        image_rgb = read_image_rgb(image_path)
                    except ValueError:
                        skipped += 1
                        continue

                    augmented_image = transform(image=image_rgb)["image"]
                    target_name = f"{image_path.stem}_srcbal{index:04d}.jpg"
                    target_path = output_dir / "train" / class_name / target_name
                    write_image_rgb(target_path, augmented_image)
                    source_generated += 1
                    generated += 1
                    augmented += 1

                source_stats[source_name] = {
                    "original": source_original_count,
                    "generated": source_generated,
                    "final": source_original_count + source_generated if keep_original else source_generated,
                }
        else:
            needed = max(0, resolved_target_count - original_count)

            for index in range(1, needed + 1):
                image_path = rng.choice(class_images)
                try:
                    image_rgb = read_image_rgb(image_path)
                except ValueError:
                    skipped += 1
                    continue

                augmented_image = transform(image=image_rgb)["image"]
                target_name = f"{image_path.stem}_bal{index:04d}.jpg"
                target_path = output_dir / "train" / class_name / target_name
                write_image_rgb(target_path, augmented_image)
                generated += 1
                augmented += 1

        class_stats[class_name] = {
            "original": original_count,
            "generated": generated,
            "final": original_count + generated if keep_original else generated,
        }
        if source_stats:
            class_stats[class_name]["sources"] = source_stats

    report_path = write_balance_report(output_dir, class_stats)

    return {
        "input_train": len(train_images),
        "copied_train": copied_train,
        "copied_val": copied_val,
        "target_count": resolved_target_count,
        "balance_sources": balance_sources,
        "augmented": augmented,
        "skipped": skipped,
        "output_dir": output_dir,
        "report_path": report_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Augment YOLO classification train crops and keep validation unchanged."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copies-per-image", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["copies", "balance"],
        default="balance",
        help="copies: fixed copies per image, balance: augment minority classes to target count.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Target train image count per class for balance mode. Defaults to the largest class count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-keep-original", action="store_true")
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--vertical-flip", action="store_true")
    parser.add_argument(
        "--balance-sources",
        action="store_true",
        help="In balance mode, also match final counts across available sources inside each class.",
    )
    parser.add_argument(
        "--source-name",
        action="append",
        default=["kaggle", "internet"],
        help="Known source filename prefix. Can be passed multiple times.",
    )
    args = parser.parse_args()

    if args.mode == "balance":
        stats = balance_train_split(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_count=args.target_count,
            seed=args.seed,
            keep_original=not args.no_keep_original,
            enable_horizontal_flip=args.horizontal_flip,
            enable_vertical_flip=args.vertical_flip,
            balance_sources=args.balance_sources,
            source_names=args.source_name,
        )
    else:
        stats = augment_train_split(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            copies_per_image=args.copies_per_image,
            seed=args.seed,
            keep_original=not args.no_keep_original,
            enable_horizontal_flip=args.horizontal_flip,
            enable_vertical_flip=args.vertical_flip,
        )

    print("=" * 60)
    print("YOLO classification crop augmentation complete")
    print("=" * 60)
    print(f"Input train      : {stats['input_train']}")
    print(f"Copied train     : {stats['copied_train']}")
    print(f"Copied val       : {stats['copied_val']}")
    if "target_count" in stats:
        print(f"Target per class : {stats['target_count']}")
        print(f"Balance sources  : {stats['balance_sources']}")
    print(f"Augmented train  : {stats['augmented']}")
    print(f"Skipped          : {stats['skipped']}")
    print(f"Output dir       : {stats['output_dir']}")
    if "report_path" in stats:
        print(f"Balance report   : {stats['report_path']}")


if __name__ == "__main__":
    main()
