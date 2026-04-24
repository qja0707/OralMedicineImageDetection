import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


IMAGE_LEVEL_KEYS = [
    "file_name",
    "width",
    "height",
    "back_color",
    "light_color",
    "camera_la",
    "camera_lo",
    "size",
    "img_regist_ts",
]

ANNOTATION_META_KEYS = [
    "drug_N",
    "drug_S",
    "drug_dir",
    "dl_idx",
    "dl_mapping_code",
    "dl_name",
    "dl_name_en",
    "img_key",
    "dl_material",
    "dl_material_en",
    "dl_custom_shape",
    "dl_company",
    "dl_company_en",
    "di_company_mf",
    "di_company_mf_en",
    "item_seq",
    "di_item_permit_date",
    "di_class_no",
    "di_etc_otc_code",
    "di_edi_code",
    "chart",
    "drug_shape",
    "thick",
    "leng_long",
    "leng_short",
    "print_front",
    "print_back",
    "color_class1",
    "color_class2",
    "line_front",
    "line_back",
    "form_code_name",
    "mark_code_front_anal",
    "mark_code_back_anal",
    "mark_code_front_img",
    "mark_code_back_img",
    "mark_code_front",
    "mark_code_back",
    "change_date",
]


def build_merged_coco(
    train_annotations_path,
    train_images_path,
    output_dir,
    output_filename="train_coco.json",
    mapping_filename="category_mapping.json",
    verbose=True,
):
    train_annotations_path = Path(train_annotations_path)
    train_images_path = Path(train_images_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_annotations_path.exists():
        raise FileNotFoundError(f"Annotation path not found: {train_annotations_path}")
    if not train_images_path.exists():
        raise FileNotFoundError(f"Image path not found: {train_images_path}")

    def log(message):
        if verbose:
            print(message)

    train_jsons = sorted(
        glob.glob(str(train_annotations_path / "**" / "*.json"), recursive=True)
    )
    train_image_files = sorted(path for path in train_images_path.rglob("*") if path.is_file())

    log("=" * 55)
    log("1단계: 원본 annotation/image 경로 읽는 중...")
    log("=" * 55)
    log(f"  발견한 train JSON : {len(train_jsons)}개")
    log(f"  발견한 train 이미지: {len(train_image_files)}개")

    if not train_jsons:
        raise ValueError(f"No annotation json files found under: {train_annotations_path}")
    if not train_image_files:
        raise ValueError(f"No image files found under: {train_images_path}")

    log("\n" + "=" * 55)
    log("2단계: 카테고리 수집 & ID 매핑 생성 중...")
    log("=" * 55)

    name_to_dlid = {}
    dlid_to_name = {}

    for json_file in train_jsons:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        for category in data.get("categories", []):
            name = category.get("name", "").strip()
            category_id = category.get("id")
            if name and category_id is not None:
                name_to_dlid[name] = category_id
                dlid_to_name[category_id] = name

    log(f"  고유 카테고리 수  : {len(name_to_dlid)}개")

    mapping = {
        "description": "category_name ↔ original category_id 매핑 테이블",
        "total_categories": len(name_to_dlid),
        "name_to_original_id": name_to_dlid,
        "original_id_to_name": {str(key): value for key, value in dlid_to_name.items()},
    }

    mapping_path = output_dir / mapping_filename
    with open(mapping_path, "w", encoding="utf-8") as file:
        json.dump(mapping, file, ensure_ascii=False, indent=2)
    log(f"  매핑 테이블 저장  : {mapping_path}")

    log("\n" + "=" * 55)
    log("3단계: 이미지 ID 매핑 생성 중...")
    log("=" * 55)

    filename_to_imageid = {}
    for index, image_path in enumerate(train_image_files, start=1):
        filename_to_imageid[image_path.name] = index

    log(f"  총 이미지 수      : {len(filename_to_imageid)}개")

    log("\n" + "=" * 55)
    log("4단계: JSON 파싱 & COCO 통합 JSON 조립 중...")
    log("=" * 55)

    coco_annotations = []
    annotation_metadata = []
    seen_image_ids = set()
    image_id_info = {}

    annotation_id_counter = 1
    skip_count = 0
    bbox_invalid = 0
    meta_missing = 0
    meta_duplicate = 0

    for json_file in train_jsons:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        images_in_json = data.get("images", [])
        if not images_in_json:
            skip_count += 1
            continue

        img_info = images_in_json[0]
        file_name = img_info.get("file_name", "")
        fname_only = Path(file_name).name

        if fname_only in filename_to_imageid:
            image_id = filename_to_imageid[fname_only]
        else:
            image_id = len(filename_to_imageid) + len(seen_image_ids) + 1
            filename_to_imageid[fname_only] = image_id

        if image_id not in seen_image_ids:
            seen_image_ids.add(image_id)
            image_id_info[image_id] = {
                "id": image_id,
                "file_name": fname_only,
                "width": img_info.get("width", 0),
                "height": img_info.get("height", 0),
                "imgfile": img_info.get("imgfile", fname_only),
                "back_color": img_info.get("back_color", ""),
                "light_color": img_info.get("light_color", ""),
                "camera_la": img_info.get("camera_la"),
                "camera_lo": img_info.get("camera_lo"),
                "size": img_info.get("size"),
                "img_regist_ts": img_info.get("img_regist_ts", ""),
            }

        annotation_meta_map = defaultdict(list)
        for raw_meta in images_in_json:
            raw_dlid = raw_meta.get("dl_idx")
            try:
                meta_category_id = int(raw_dlid)
            except (TypeError, ValueError):
                continue
            annotation_meta_map[meta_category_id].append(raw_meta)

        for ann in data.get("annotations", []):
            bbox = ann.get("bbox")
            category_id = ann.get("category_id")
            area = ann.get("area")
            iscrowd = ann.get("iscrowd", 0)
            segmentation = ann.get("segmentation", [])
            ignore = ann.get("ignore", 0)

            if not isinstance(bbox, list) or len(bbox) != 4:
                bbox_invalid += 1
                continue

            if all(value == 0 for value in bbox):
                bbox_invalid += 1
                continue

            coco_annotations.append(
                {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": iscrowd,
                    "segmentation": segmentation,
                    "ignore": ignore,
                }
            )

            matched_meta = None
            candidates = annotation_meta_map.get(category_id, [])
            if candidates:
                matched_meta = candidates.pop(0)
                if candidates:
                    meta_duplicate += len(candidates)
            else:
                meta_missing += 1

            meta_record = {
                "annotation_id": annotation_id_counter,
                "image_id": image_id,
                "category_id": category_id,
                "file_name": fname_only,
            }

            for key in IMAGE_LEVEL_KEYS:
                meta_record[key] = img_info.get(key)

            for key in ANNOTATION_META_KEYS:
                meta_record[key] = matched_meta.get(key) if matched_meta else None

            annotation_metadata.append(meta_record)
            annotation_id_counter += 1

    coco_images = sorted(image_id_info.values(), key=lambda item: item["id"])
    coco_categories = [
        {
            "id": original_id,
            "name": name,
            "supercategory": "pill",
        }
        for original_id, name in sorted(dlid_to_name.items(), key=lambda item: item[0])
    ]

    log(f"  처리된 이미지 수  : {len(coco_images)}개")
    log(f"  유효 어노테이션   : {len(coco_annotations)}개")
    log(f"  무효/스킵 수      : {skip_count + bbox_invalid}개")
    log(f"  메타 누락 수      : {meta_missing}개")
    log(f"  메타 중복 후보 수 : {meta_duplicate}개")

    log("\n" + "=" * 55)
    log("5단계: 통합 COCO JSON 저장 중...")
    log("=" * 55)

    coco_output = {
        "info": {
            "description": "알약 탐지 프로젝트 통합 어노테이션",
            "version": "1.0",
            "total_images": len(coco_images),
            "total_annotations": len(coco_annotations),
            "total_categories": len(coco_categories),
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    coco_output_path = output_dir / output_filename
    with open(coco_output_path, "w", encoding="utf-8") as file:
        json.dump(coco_output, file, ensure_ascii=False, indent=2)
    log(f"  저장 완료         : {coco_output_path}")

    annotation_metadata_path = output_dir / "annotation_metadata.json"
    with open(annotation_metadata_path, "w", encoding="utf-8") as file:
        json.dump(annotation_metadata, file, ensure_ascii=False, indent=2)
    log(f"  메타 저장 완료    : {annotation_metadata_path}")

    class_ann_count = defaultdict(int)
    for ann in coco_annotations:
        class_ann_count[ann["category_id"]] += 1

    img_pill_count = defaultdict(int)
    for ann in coco_annotations:
        img_pill_count[ann["image_id"]] += 1

    pill_counts = list(img_pill_count.values())
    avg_per_image = sum(pill_counts) / len(pill_counts) if pill_counts else 0
    dist = defaultdict(int)
    for count in pill_counts:
        dist[count] += 1

    report_lines = [
        "=" * 55,
        "  알약 탐지 데이터셋 현황 요약",
        "=" * 55,
        f"  Train 이미지 수        : {len(coco_images)}장",
        f"  Train 어노테이션 수    : {len(coco_annotations)}개",
        f"  카테고리(약품) 수      : {len(coco_categories)}종",
        f"  이미지당 평균 알약 수  : {avg_per_image:.2f}개",
        f"  메타 누락 수           : {meta_missing}개",
        f"  메타 중복 후보 수      : {meta_duplicate}개",
        "",
        "  이미지당 알약 수 분포",
        *[f"    알약 {key}개짜리 이미지  : {value}장" for key, value in sorted(dist.items())],
        "",
        "  클래스별 어노테이션 수 (적은 순)",
        *[
            f"    [{cat_id:>6}] {dlid_to_name.get(cat_id, 'UNKNOWN'):<35} : {class_ann_count[cat_id]}개"
            for cat_id in sorted(class_ann_count, key=lambda item: class_ann_count[item])
        ],
    ]

    if class_ann_count:
        report_lines.extend(
            [
                "",
                f"  최소 클래스 어노테이션 : {min(class_ann_count.values())}개",
                f"  최대 클래스 어노테이션 : {max(class_ann_count.values())}개",
                f"  평균 클래스 어노테이션 : {sum(class_ann_count.values()) / len(class_ann_count):.1f}개",
            ]
        )

    report_lines.append("=" * 55)
    report_text = "\n".join(report_lines)

    log("\n" + "=" * 55)
    log("6단계: 데이터 현황 리포트 생성 중...")
    log("=" * 55)
    log(report_text)

    summary_path = output_dir / "dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(report_text)
    log(f"\n  리포트 저장       : {summary_path}")

    log("\n" + "=" * 55)
    log("  완료! 생성된 파일 목록")
    log("=" * 55)
    log(f"  1. {coco_output_path}")
    log(f"  2. {annotation_metadata_path}")
    log(f"  3. {mapping_path}")
    log(f"  4. {summary_path}")
    log("=" * 55)

    return {
        "coco": coco_output,
        "mapping": mapping,
        "annotation_metadata": annotation_metadata,
        "paths": {
            "coco_output_path": coco_output_path,
            "mapping_path": mapping_path,
            "annotation_metadata_path": annotation_metadata_path,
            "summary_path": summary_path,
        },
        "stats": {
            "num_json_files": len(train_jsons),
            "num_image_files": len(train_image_files),
            "num_images": len(coco_images),
            "num_annotations": len(coco_annotations),
            "num_categories": len(coco_categories),
            "meta_missing": meta_missing,
            "meta_duplicate": meta_duplicate,
            "invalid_or_skipped": skip_count + bbox_invalid,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Merge raw annotations into a single COCO json.")
    parser.add_argument("--train-annotations-path", required=True)
    parser.add_argument("--train-images-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-filename", default="train_coco.json")
    parser.add_argument("--mapping-filename", default="category_mapping.json")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logging.",
    )
    args = parser.parse_args()

    build_merged_coco(
        train_annotations_path=args.train_annotations_path,
        train_images_path=args.train_images_path,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        mapping_filename=args.mapping_filename,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
