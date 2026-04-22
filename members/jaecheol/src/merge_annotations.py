import json
import zipfile
from pathlib import Path
from collections import defaultdict


# 데이터 로드
ZIP_PATH   = r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\ai10-level1-project.zip"
OUTPUT_DIR = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 파일 목록 확인
print("=" * 55)
print("1단계: ZIP 파일 읽는 중...")
print("=" * 55)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    all_files = z.namelist()

train_jsons = [
    f for f in all_files
    if f.endswith(".json") and "train_annotations" in f
]
train_pngs = [
    f for f in all_files
    if f.endswith(".png") and "train_images" in f
]

print(f"  발견한 train JSON : {len(train_jsons)}개")
print(f"  발견한 train PNG  : {len(train_pngs)}개")


# 카테고리 수집 및 순차적으로 ID 부여 후 매핑
print("\n" + "=" * 55)
print("2단계: 카테고리 수집 & ID 매핑 생성 중...")
print("=" * 55)

# 카테고리명 → dl_idx(원본 숫자 ID) 수집
#   JSON 구조: categories[0].name = 약품명, categories[0].id = dl_idx
name_to_dlid: dict[str, int] = {}   # 약품명 → 원본 dl_idx
dlid_to_name: dict[int, str] = {}   # 원본 dl_idx → 약품명

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    for jf in train_jsons:
        with z.open(jf) as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            name = cat.get("name", "").strip()
            cid  = cat.get("id")
            if name and cid is not None:
                name_to_dlid[name] = cid
                dlid_to_name[cid]  = name

# 카테고리명 기준으로 정렬 후 1-based 순차 ID 부여
sorted_names = sorted(name_to_dlid.keys())
name_to_newid: dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(sorted_names)
}
newid_to_name: dict[int, str] = {v: k for k, v in name_to_newid.items()}
dlid_to_newid: dict[int, int] = {
    name_to_dlid[name]: name_to_newid[name]
    for name in sorted_names
    if name in name_to_dlid
}

print(f"  고유 카테고리 수  : {len(sorted_names)}개")
print(f"  ID 범위           : 1 ~ {len(sorted_names)}")

# 매핑 테이블 저장
mapping = {
    "description": "category_name ↔ category_id 매핑 테이블",
    "total_categories": len(sorted_names),
    "name_to_id": name_to_newid,
    "id_to_name": {str(k): v for k, v in newid_to_name.items()},
    "original_dlid_to_new_id": {str(k): v for k, v in dlid_to_newid.items()},
}
mapping_path = OUTPUT_DIR / "category_mapping.json"
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print(f"  매핑 테이블 저장  : {mapping_path}")


# 이미지 파일명 및 이미지 ID 매핑
print("\n" + "=" * 55)
print("3단계: 이미지 ID 매핑 생성 중...")
print("=" * 55)

# train_images 폴더 안의 PNG 파일명으로 image_id 결정
# 파일명 예: K-001900-016548-019607-029451_0_2_0_2_70_000_200.png
# image_id = 파일명의 숫자 부분 (파일명 자체를 hash 또는 순번으로)

filename_to_imageid: dict[str, int] = {}
imageid_to_filename: dict[int, str] = {}

for idx, png_path in enumerate(sorted(train_pngs), start=1):
    fname = Path(png_path).name   # 예: K-001900-..._200.png
    filename_to_imageid[fname] = idx
    imageid_to_filename[idx]   = fname

print(f"  총 이미지 수      : {len(filename_to_imageid)}개")


#전체 JSON 파싱 → COCO 통합 구조
print("\n" + "=" * 55)
print("4단계: JSON 763개 파싱 & COCO 통합 JSON 조립 중...")
print("=" * 55)

coco_images:      list[dict] = []
coco_annotations: list[dict] = []

seen_image_ids:  set[int] = set()
annotation_id_counter = 1
skip_count   = 0
bbox_invalid = 0

# 이미 수집한 image 정보를 JSON에서도 읽어둠
image_id_info: dict[int, dict] = {}  # image_id → {width, height, file_name}

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    for jf in train_jsons:
        with z.open(jf) as f:
            data = json.load(f)

        # 이미지 정보
        images_in_json = data.get("images", [])
        if not images_in_json:
            skip_count += 1
            continue

        img_info  = images_in_json[0]
        file_name = img_info.get("file_name", "")

        # image_id 결정: 파일명으로 찾거나 없으면 신규 부여
        fname_only = Path(file_name).name
        if fname_only in filename_to_imageid:
            image_id = filename_to_imageid[fname_only]
        else:
            # 파일명이 다를 경우 순번으로 처리
            image_id = len(filename_to_imageid) + len(seen_image_ids) + 1
            filename_to_imageid[fname_only] = image_id

        # 이미지 중복 등록 방지
        if image_id not in seen_image_ids:
            seen_image_ids.add(image_id)
            image_id_info[image_id] = {
                "id"       : image_id,
                "file_name": fname_only,
                "width"    : img_info.get("width", 0),
                "height"   : img_info.get("height", 0),
            }

        # 어노테이션
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox")

            # 유효성 검사: bbox 존재 & 길이 4
            if not isinstance(bbox, list) or len(bbox) != 4:
                bbox_invalid += 1
                continue

            # bbox 값이 모두 0인 경우 제외 (빈 bbox)
            if all(v == 0 for v in bbox):
                bbox_invalid += 1
                continue

            # category_id 변환: 원본 dl_idx → 새 순차 ID
            orig_cat_id = ann.get("category_id")
            new_cat_id  = dlid_to_newid.get(orig_cat_id)
            if new_cat_id is None:
                # 카테고리명 기반 재시도
                cats = data.get("categories", [])
                if cats:
                    cat_name   = cats[0].get("name", "").strip()
                    new_cat_id = name_to_newid.get(cat_name)
            if new_cat_id is None:
                skip_count += 1
                continue

            # bbox: [x, y, w, h] → 면적 계산
            x, y, w, h = [float(v) for v in bbox]
            area = w * h

            coco_annotations.append({
                "id"         : annotation_id_counter,
                "image_id"   : image_id,
                "category_id": new_cat_id,
                "bbox"       : [x, y, w, h],
                "area"       : round(area, 2),
                "iscrowd"    : 0,
                "segmentation": [],
            })
            annotation_id_counter += 1

# 이미지 리스트 정렬
coco_images = sorted(image_id_info.values(), key=lambda x: x["id"])

print(f"  처리된 이미지 수  : {len(coco_images)}개")
print(f"  유효 어노테이션   : {len(coco_annotations)}개")
print(f"  무효/스킵 수      : {skip_count + bbox_invalid}개")


# 5단계: COCO categories 리스트 생성 
coco_categories = [
    {
        "id"           : new_id,
        "name"         : name,
        "supercategory": "pill",
    }
    for name, new_id in sorted(name_to_newid.items(), key=lambda x: x[1])
]


# 6단계: 통합 COCO JSON 저장 
print("\n" + "=" * 55)
print("5단계: 통합 COCO JSON 저장 중...")
print("=" * 55)

coco_output = {
    "info": {
        "description": "알약 탐지 프로젝트 통합 어노테이션",
        "version"    : "1.0",
        "total_images"     : len(coco_images),
        "total_annotations": len(coco_annotations),
        "total_categories" : len(coco_categories),
    },
    "images"     : coco_images,
    "annotations": coco_annotations,
    "categories" : coco_categories,
}

output_path = OUTPUT_DIR / "train_coco.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(coco_output, f, ensure_ascii=False, indent=2)
print(f"  저장 완료         : {output_path}")


# 7단계: 데이터 현황 요약 리포트 
print("\n" + "=" * 55)
print("6단계: 데이터 현황 리포트 생성 중...")
print("=" * 55)

# 클래스별 어노테이션 수 집계
class_ann_count: dict[int, int] = defaultdict(int)
for ann in coco_annotations:
    class_ann_count[ann["category_id"]] += 1

# 이미지당 알약 수 집계
img_pill_count: dict[int, int] = defaultdict(int)
for ann in coco_annotations:
    img_pill_count[ann["image_id"]] += 1

pill_counts   = list(img_pill_count.values())
avg_per_image = sum(pill_counts) / len(pill_counts) if pill_counts else 0
dist          = defaultdict(int)
for c in pill_counts:
    dist[c] += 1

report_lines = [
    "=" * 55,
    "  알약 탐지 데이터셋 현황 요약",
    "=" * 55,
    f"  Train 이미지 수        : {len(coco_images)}장",
    f"  Train 어노테이션 수    : {len(coco_annotations)}개",
    f"  카테고리(약품) 수      : {len(coco_categories)}종",
    f"  이미지당 평균 알약 수  : {avg_per_image:.2f}개",
    "",
    "  이미지당 알약 수 분포",
    *[f"    알약 {k}개짜리 이미지  : {v}장" for k, v in sorted(dist.items())],
    "",
    "  클래스별 어노테이션 수 (적은 순)",
    *[
        f"    [{new_id:>3}] {newid_to_name[new_id]:<35} : {class_ann_count[new_id]}개"
        for new_id in sorted(class_ann_count, key=lambda x: class_ann_count[x])
    ],
    "",
    f"  최소 클래스 어노테이션 : {min(class_ann_count.values())}개",
    f"  최대 클래스 어노테이션 : {max(class_ann_count.values())}개",
    f"  평균 클래스 어노테이션 : {sum(class_ann_count.values())/len(class_ann_count):.1f}개",
    "=" * 55,
]

report_text = "\n".join(report_lines)
print(report_text)

summary_path = OUTPUT_DIR / "dataset_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n  리포트 저장       : {summary_path}")

print("\n" + "=" * 55)
print("  완료! 생성된 파일 목록")
print("=" * 55)
print(f"  1. {OUTPUT_DIR / 'train_coco.json'}")
print(f"  2. {OUTPUT_DIR / 'category_mapping.json'}")
print(f"  3. {OUTPUT_DIR / 'dataset_summary.txt'}")
print("=" * 55)
