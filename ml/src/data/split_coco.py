import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def default_group_key(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.split("_", 1)[0]


def load_coco(coco_json_path):
    coco_json_path = Path(coco_json_path)
    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO json not found: {coco_json_path}")

    with open(coco_json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_group_records(coco):
    images_by_id = {image["id"]: image for image in coco["images"]}
    annotations_by_image = defaultdict(list)
    class_image_count = defaultdict(set)

    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        annotations_by_image[image_id].append(annotation)
        class_image_count[annotation["category_id"]].add(image_id)

    groups = {}
    for image in coco["images"]:
        group_id = default_group_key(image["file_name"])
        group = groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "image_ids": [],
                "category_counts": Counter(),
            },
        )
        group["image_ids"].append(image["id"])
        for annotation in annotations_by_image.get(image["id"], []):
            group["category_counts"][annotation["category_id"]] += 1

    class_group_count = defaultdict(int)
    for group in groups.values():
        for category_id in group["category_counts"]:
            class_group_count[category_id] += 1

    for group in groups.values():
        rarity = min(class_group_count[category_id] for category_id in group["category_counts"])
        group["rarity"] = rarity
        group["num_images"] = len(group["image_ids"])
        group["num_annotations"] = sum(group["category_counts"].values())

    class_image_count = {category_id: len(image_ids) for category_id, image_ids in class_image_count.items()}
    return groups, annotations_by_image, images_by_id, class_image_count, dict(class_group_count)


def find_single_group_category_groups(groups, class_group_count):
    forced_group_ids = set()
    forced_category_ids = set()

    for group in groups:
        for category_id in group["category_counts"]:
            if class_group_count[category_id] == 1:
                forced_group_ids.add(group["group_id"])
                forced_category_ids.add(category_id)

    return forced_group_ids, forced_category_ids


def build_forced_val_category_rows(
    category_ids,
    categories,
    class_image_count,
    class_group_count,
    groups,
):
    category_name_map = {category["id"]: category["name"] for category in categories}
    category_to_group_ids = defaultdict(list)
    category_to_image_count_in_forced_groups = defaultdict(int)

    for group in groups:
        for category_id in group["category_counts"]:
            if category_id in category_ids:
                category_to_group_ids[category_id].append(group["group_id"])
                category_to_image_count_in_forced_groups[category_id] += group["num_images"]

    rows = []
    for category_id in sorted(category_ids):
        rows.append(
            {
                "category_id": category_id,
                "category_name": category_name_map.get(category_id, "UNKNOWN"),
                "num_groups": class_group_count.get(category_id, 0),
                "num_images": class_image_count.get(category_id, 0),
                "forced_val_group_ids": category_to_group_ids.get(category_id, []),
                "forced_val_image_count": category_to_image_count_in_forced_groups.get(category_id, 0),
            }
        )
    return rows


def assign_validation_groups(groups, val_image_target, seed, forced_val_group_ids=None):
    random_generator = random.Random(seed)
    groups_by_id = {group["group_id"]: group for group in groups}
    assigned_val_groups = set(forced_val_group_ids or [])
    current_val_images = sum(
        groups_by_id[group_id]["num_images"]
        for group_id in assigned_val_groups
        if group_id in groups_by_id
    )

    class_to_groups = defaultdict(list)
    for group in groups:
        for category_id in group["category_counts"]:
            class_to_groups[category_id].append(group["group_id"])

    ordered_categories = sorted(
        class_to_groups,
        key=lambda category_id: (len(class_to_groups[category_id]), category_id),
    )

    for category_id in ordered_categories:
        if current_val_images >= val_image_target:
            break

        candidates = [
            groups_by_id[group_id]
            for group_id in class_to_groups[category_id]
            if group_id not in assigned_val_groups
        ]
        if not candidates:
            continue

        candidates.sort(
            key=lambda group: (
                group["rarity"],
                group["num_images"],
                -group["num_annotations"],
                group["group_id"],
            )
        )
        chosen_group = candidates[0]
        assigned_val_groups.add(chosen_group["group_id"])
        current_val_images += chosen_group["num_images"]

    remaining_groups = [
        group for group in groups if group["group_id"] not in assigned_val_groups
    ]
    random_generator.shuffle(remaining_groups)
    remaining_groups.sort(
        key=lambda group: (
            max(0, val_image_target - current_val_images - group["num_images"]),
            group["num_images"],
        )
    )

    for group in remaining_groups:
        if current_val_images >= val_image_target:
            break
        assigned_val_groups.add(group["group_id"])
        current_val_images += group["num_images"]

    return assigned_val_groups


def build_split_coco(coco, target_image_ids):
    target_image_ids = set(target_image_ids)
    split_images = [image for image in coco["images"] if image["id"] in target_image_ids]
    split_annotations = [
        annotation
        for annotation in coco["annotations"]
        if annotation["image_id"] in target_image_ids
    ]

    return {
        "info": coco.get("info", {}),
        "images": split_images,
        "annotations": split_annotations,
        "categories": coco["categories"],
    }


def split_coco_dataset(
    coco_json_path,
    output_dir,
    train_filename="train_coco.json",
    val_filename="val_coco.json",
    split_metadata_filename="train_val_split.json",
    forced_val_categories_filename="forced_val_single_group_categories.csv",
    val_ratio=0.2,
    seed=42,
    force_single_group_classes_to_val=True,
    verbose=True,
):
    coco = load_coco(coco_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def log(message):
        if verbose:
            print(message)

    groups_by_id, annotations_by_image, images_by_id, class_image_count, class_group_count = build_group_records(coco)
    groups = sorted(groups_by_id.values(), key=lambda group: group["group_id"])

    total_images = len(coco["images"])
    total_annotations = len(coco["annotations"])
    val_image_target = max(1, round(total_images * val_ratio))

    log("=" * 55)
    log("1단계: split 대상 그룹 구성 중...")
    log("=" * 55)
    log(f"  총 이미지 수      : {total_images}개")
    log(f"  총 어노테이션 수  : {total_annotations}개")
    log(f"  그룹 수           : {len(groups)}개")
    log(f"  val 목표 이미지 수: {val_image_target}개")

    forced_val_group_ids = set()
    forced_val_category_ids = set()
    if force_single_group_classes_to_val:
        forced_val_group_ids, forced_val_category_ids = find_single_group_category_groups(
            groups,
            class_group_count,
        )
        forced_val_image_count = sum(
            groups_by_id[group_id]["num_images"] for group_id in forced_val_group_ids
        )
        log(f"  단일 그룹 클래스 수: {len(forced_val_category_ids)}개")
        log(f"  단일 그룹 val 강제 그룹 수: {len(forced_val_group_ids)}개")
        log(f"  단일 그룹 val 강제 이미지 수: {forced_val_image_count}개")

    forced_val_category_rows = build_forced_val_category_rows(
        forced_val_category_ids,
        coco["categories"],
        class_image_count,
        class_group_count,
        groups,
    )

    if forced_val_category_rows:
        log("  단일 그룹 val 강제 클래스 예시:")
        for row in forced_val_category_rows[:20]:
            log(
                f"    [{row['category_id']}] {row['category_name']} "
                f"(groups={row['num_groups']}, images={row['num_images']})"
            )

    val_group_ids = assign_validation_groups(
        groups,
        val_image_target,
        seed,
        forced_val_group_ids=forced_val_group_ids,
    )

    train_image_ids = []
    val_image_ids = []
    train_group_ids = []
    val_group_ids_sorted = []

    for group in groups:
        if group["group_id"] in val_group_ids:
            val_group_ids_sorted.append(group["group_id"])
            val_image_ids.extend(group["image_ids"])
        else:
            train_group_ids.append(group["group_id"])
            train_image_ids.extend(group["image_ids"])

    train_coco = build_split_coco(coco, train_image_ids)
    val_coco = build_split_coco(coco, val_image_ids)

    train_class_count = Counter()
    val_class_count = Counter()
    for annotation in train_coco["annotations"]:
        train_class_count[annotation["category_id"]] += 1
    for annotation in val_coco["annotations"]:
        val_class_count[annotation["category_id"]] += 1

    missing_in_val = sorted(
        category["id"]
        for category in coco["categories"]
        if train_class_count[category["id"]] > 0 and val_class_count[category["id"]] == 0
    )
    missing_in_train = sorted(
        category["id"]
        for category in coco["categories"]
        if val_class_count[category["id"]] > 0 and train_class_count[category["id"]] == 0
    )

    log("\n" + "=" * 55)
    log("2단계: train/val split 결과 집계 중...")
    log("=" * 55)
    log(f"  Train 그룹 수      : {len(train_group_ids)}개")
    log(f"  Val 그룹 수        : {len(val_group_ids_sorted)}개")
    log(f"  Train 이미지 수    : {len(train_coco['images'])}개")
    log(f"  Val 이미지 수      : {len(val_coco['images'])}개")
    log(f"  Train 어노테이션 수: {len(train_coco['annotations'])}개")
    log(f"  Val 어노테이션 수  : {len(val_coco['annotations'])}개")
    log(f"  Val 누락 클래스 수 : {len(missing_in_val)}개")
    log(f"  Train 누락 클래스 수: {len(missing_in_train)}개")

    category_name_map = {category["id"]: category["name"] for category in coco["categories"]}
    if missing_in_val:
        preview = ", ".join(category_name_map[category_id] for category_id in missing_in_val[:10])
        log(f"  누락 클래스 예시   : {preview}")
    if missing_in_train:
        preview = ", ".join(category_name_map[category_id] for category_id in missing_in_train[:10])
        log(f"  Train 누락 클래스 예시: {preview}")

    train_output_path = output_dir / train_filename
    val_output_path = output_dir / val_filename
    metadata_output_path = output_dir / split_metadata_filename
    forced_val_categories_output_path = output_dir / forced_val_categories_filename

    with open(train_output_path, "w", encoding="utf-8") as file:
        json.dump(train_coco, file, ensure_ascii=False, indent=2)
    with open(val_output_path, "w", encoding="utf-8") as file:
        json.dump(val_coco, file, ensure_ascii=False, indent=2)

    split_metadata = {
        "seed": seed,
        "val_ratio": val_ratio,
        "force_single_group_classes_to_val": force_single_group_classes_to_val,
        "grouping_rule": "file_name stem prefix before first underscore",
        "group_key_example": "K-001900-016548-019607-029451_0_2_0_2_70_000_200.png -> K-001900-016548-019607-029451",
        "num_groups": len(groups),
        "forced_val_group_ids": sorted(forced_val_group_ids),
        "forced_val_category_ids": sorted(forced_val_category_ids),
        "forced_val_single_group_categories": forced_val_category_rows,
        "train_group_ids": train_group_ids,
        "val_group_ids": val_group_ids_sorted,
        "train_image_ids": sorted(train_image_ids),
        "val_image_ids": sorted(val_image_ids),
        "class_image_count": {str(category_id): count for category_id, count in sorted(class_image_count.items())},
        "class_group_count": {str(category_id): count for category_id, count in sorted(class_group_count.items())},
        "missing_in_val_category_ids": missing_in_val,
        "missing_in_train_category_ids": missing_in_train,
    }

    with open(metadata_output_path, "w", encoding="utf-8") as file:
        json.dump(split_metadata, file, ensure_ascii=False, indent=2)

    with open(forced_val_categories_output_path, "w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "category_id",
            "category_name",
            "num_groups",
            "num_images",
            "forced_val_image_count",
            "forced_val_group_ids",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in forced_val_category_rows:
            csv_row = dict(row)
            csv_row["forced_val_group_ids"] = ",".join(row["forced_val_group_ids"])
            writer.writerow(csv_row)

    log("\n" + "=" * 55)
    log("3단계: split 산출물 저장 완료")
    log("=" * 55)
    log(f"  Train COCO         : {train_output_path}")
    log(f"  Val COCO           : {val_output_path}")
    log(f"  Split metadata     : {metadata_output_path}")
    log(f"  단일 그룹 val 클래스: {forced_val_categories_output_path}")

    return {
        "train_coco": train_coco,
        "val_coco": val_coco,
        "split_metadata": split_metadata,
        "paths": {
            "train_output_path": train_output_path,
            "val_output_path": val_output_path,
            "metadata_output_path": metadata_output_path,
            "forced_val_categories_output_path": forced_val_categories_output_path,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Split merged COCO into train/val sets.")
    parser.add_argument("--coco-json-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-filename", default="train_coco.json")
    parser.add_argument("--val-filename", default="val_coco.json")
    parser.add_argument("--split-metadata-filename", default="train_val_split.json")
    parser.add_argument(
        "--forced-val-categories-filename",
        default="forced_val_single_group_categories.csv",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-single-group-classes-in-train",
        action="store_true",
        help="Disable the default behavior that forces classes with only one source group into val.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    split_coco_dataset(
        coco_json_path=args.coco_json_path,
        output_dir=args.output_dir,
        train_filename=args.train_filename,
        val_filename=args.val_filename,
        split_metadata_filename=args.split_metadata_filename,
        forced_val_categories_filename=args.forced_val_categories_filename,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force_single_group_classes_to_val=not args.keep_single_group_classes_in_train,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
