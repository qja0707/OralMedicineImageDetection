import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np



def load_json(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def clear_split_dir(split_dir):
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)


def build_category_mappings(categories):
    sorted_categories = sorted(categories, key=lambda category: category["id"])
    coco_to_yolo = {
        category["id"]: index
        for index, category in enumerate(sorted_categories)
    }
    yolo_names = [str(category["id"]) for category in sorted_categories]
    return coco_to_yolo, yolo_names


def write_yolo_labels(split_coco, split_name, output_dir, raw_images_dir, coco_to_yolo, verbose, target_size=640):
    split_images_dir = output_dir / "images" / split_name
    split_labels_dir = output_dir / "labels" / split_name
    clear_split_dir(split_images_dir)
    clear_split_dir(split_labels_dir)

    annotations_by_image = defaultdict(list)
    for annotation in split_coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    converted = 0
    skipped = 0

    for image in split_coco["images"]:
        file_name = image["file_name"]
        source_image_path = raw_images_dir / file_name
        if not source_image_path.exists():
            skipped += 1
            continue

        # 1. 이미지 로드
        img = cv2.imread(str(source_image_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        
        # 2. 레터박스 계산 (비율 유지 리사이즈)
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # 3. 검은색 캔버스 생성 및 이미지 중앙 배치
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized

        # 4. 이미지 저장
        target_image_path = split_images_dir / f"{Path(file_name).stem}.jpg"
        cv2.imwrite(str(target_image_path), canvas)

        # 5. 라벨 변환 (레터박스 패딩값 반영)
        label_lines = []
        for annotation in annotations_by_image.get(image["id"], []):
            x, y, aw, ah = annotation["bbox"]
            yolo_class_id = coco_to_yolo[annotation["category_id"]]

            # 레터박스가 적용된 이미지에서의 새로운 중심점 계산
            # 원본 좌표 -> 리사이즈 좌표 -> 패딩 더하기 -> 640으로 나누기
            cx = (x * scale + aw * scale / 2 + pad_x) / target_size
            cy = (y * scale + ah * scale / 2 + pad_y) / target_size
            nw = (aw * scale) / target_size
            nh = (ah * scale) / target_size

            label_lines.append(f"{yolo_class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = split_labels_dir / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        converted += 1

    return {"converted": converted, "skipped": skipped}



def write_data_yaml(output_dir, yolo_names):
    yaml_lines = [
        "# YOLO dataset config",
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(yolo_names)}",
        "",
        "names:",
    ]

    for index, name in enumerate(yolo_names):
        yaml_lines.append(f"  {index}: {name}")

    yaml_path = output_dir / "pill.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def build_yolo_dataset(
    train_coco_path,
    val_coco_path,
    raw_images_dir=None,
    output_dir=None,
    train_images_dir=None,
    val_images_dir=None,
    verbose=True,
):
    train_coco = load_json(train_coco_path)
    val_coco = load_json(val_coco_path)

    if raw_images_dir is not None:
        raw_images_dir = Path(raw_images_dir)
    train_images_dir = Path(train_images_dir) if train_images_dir is not None else raw_images_dir
    val_images_dir = Path(val_images_dir) if val_images_dir is not None else raw_images_dir
    if train_images_dir is None or val_images_dir is None:
        raise ValueError("Provide either raw_images_dir or both train_images_dir and val_images_dir")
    if output_dir is None:
        raise ValueError("Provide output_dir")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    if not train_images_dir.exists():
        raise FileNotFoundError(f"Train images directory not found: {train_images_dir}")
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Val images directory not found: {val_images_dir}")

    def log(message):
        if verbose:
            print(message)

    coco_to_yolo, yolo_names = build_category_mappings(train_coco["categories"])

    log("=" * 55)
    log("1단계: split COCO 기반 YOLO 데이터셋 생성 중...")
    log("=" * 55)
    log(f"  Train 이미지 수    : {len(train_coco['images'])}개")
    log(f"  Val 이미지 수      : {len(val_coco['images'])}개")
    log(f"  클래스 수          : {len(yolo_names)}개")
    log(f"  Train 이미지 경로  : {train_images_dir}")
    log(f"  Val 이미지 경로    : {val_images_dir}")

    train_stats = write_yolo_labels(
        split_coco=train_coco,
        split_name="train",
        output_dir=output_dir,
        raw_images_dir=train_images_dir,
        coco_to_yolo=coco_to_yolo,
        verbose=verbose,
    )
    val_stats = write_yolo_labels(
        split_coco=val_coco,
        split_name="val",
        output_dir=output_dir,
        raw_images_dir=val_images_dir,
        coco_to_yolo=coco_to_yolo,
        verbose=verbose,
    )

    yaml_path = write_data_yaml(output_dir, yolo_names)

    log("\n" + "=" * 55)
    log("2단계: 변환 결과 집계 중...")
    log("=" * 55)
    log(f"  Train 변환 완료 수 : {train_stats['converted']}개")
    log(f"  Train 스킵 수      : {train_stats['skipped']}개")
    log(f"  Val 변환 완료 수   : {val_stats['converted']}개")
    log(f"  Val 스킵 수        : {val_stats['skipped']}개")
    log(f"  YAML 저장 경로     : {yaml_path}")

    return {
        "train_stats": train_stats,
        "val_stats": val_stats,
        "yaml_path": yaml_path,
        "class_names": yolo_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Build YOLO dataset from split COCO files.")
    parser.add_argument("--train-coco-path", required=True)
    parser.add_argument("--val-coco-path", required=True)
    parser.add_argument(
        "--raw-images-dir",
        help="Image directory shared by train and val. Kept for backward compatibility.",
    )
    parser.add_argument("--train-images-dir", help="Train image directory.")
    parser.add_argument("--val-images-dir", help="Val image directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    build_yolo_dataset(
        train_coco_path=args.train_coco_path,
        val_coco_path=args.val_coco_path,
        output_dir=args.output_dir,
        raw_images_dir=args.raw_images_dir,
        train_images_dir=args.train_images_dir,
        val_images_dir=args.val_images_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
