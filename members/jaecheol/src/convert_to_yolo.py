"""
COCO JSON → YOLO 포맷 변환
-----------------------------------
입력: output/train_coco.json, output/category_mapping.json
      (+ ZIP 내 train_images PNG)
출력:
  dataset/
    images/train/*.png
    images/val/*.png
    labels/train/*.txt
    labels/val/*.txt
  pill.yaml
"""

import json
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

import random
random.seed(42)

# 경로 설정
ZIP_PATH      = r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\ai10-level1-project.zip"
OUTPUT_DIR    = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\output")
COCO_JSON     = OUTPUT_DIR / "train_coco.json"
MAPPING_JSON  = OUTPUT_DIR / "category_mapping.json"
DATASET_DIR   = OUTPUT_DIR / "dataset"
VAL_RATIO     = 0.2   # 80/20 split

# 데이터셋 폴더 생성
for split in ["train", "val"]:
    (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# COCO JSON 로드 
print("=" * 55)
print("COCO JSON → YOLO 포맷 변환 시작")
print("=" * 55)

with open(COCO_JSON, encoding="utf-8") as f:
    coco = json.load(f)

with open(MAPPING_JSON, encoding="utf-8") as f:
    mapping = json.load(f)

# 카테고리 정보 (0-based 변환용)
id_to_name = {int(k): v for k, v in mapping["id_to_name"].items()}
# COCO category_id(1-based) → YOLO class_id(0-based)
coco_id_to_yolo = {cat_id: cat_id - 1 for cat_id in id_to_name}

# 이미지 정보 딕셔너리
images = {img["id"]: img for img in coco["images"]}

# 이미지별 어노테이션 그룹핑
img_anns: dict[int, list] = defaultdict(list)
for ann in coco["annotations"]:
    img_anns[ann["image_id"]].append(ann)

# Stratified Train/Val Split 
# 각 클래스가 val에 최소 1개 이상 들어가도록 분할
print("\n[1] Stratified train/val split 진행 중...")

# 클래스별 이미지 ID 수집
class_to_imgs: dict[int, set] = defaultdict(set)
for ann in coco["annotations"]:
    class_to_imgs[ann["category_id"]].add(ann["image_id"])

all_img_ids = list(images.keys())
random.shuffle(all_img_ids)

# 전체 80/20 비율 기준으로 val 수 고정
n_total = len(all_img_ids)
n_val   = max(1, round(n_total * VAL_RATIO))   # 약 46장
n_train = n_total - n_val                       # 약 186장

# 클래스별로 val에 최소 1장 보장하면서 총 n_val을 넘지 않도록
val_img_ids: set[int] = set()

# 1단계: 소수 클래스(3개짜리)부터 val에 1장씩 배정
sorted_classes = sorted(class_to_imgs.items(), key=lambda x: len(x[1]))
for cat_id, img_ids in sorted_classes:
    if len(val_img_ids) >= n_val:
        break
    # 아직 val에 없는 이미지 중 1장만 배정
    candidates = list(img_ids - val_img_ids)
    if candidates:
        val_img_ids.add(random.choice(candidates))

# 2단계: 나머지 val 슬롯을 랜덤으로 채움 (총 n_val까지)
remaining = [i for i in all_img_ids if i not in val_img_ids]
random.shuffle(remaining)
still_need = n_val - len(val_img_ids)
val_img_ids.update(remaining[:still_need])

train_img_ids = set(all_img_ids) - val_img_ids

print(f"  Train 이미지: {len(train_img_ids)}장")
print(f"  Val   이미지: {len(val_img_ids)}장")

# YOLO txt 파일 생성 + 이미지 복사
print("\n[2] YOLO txt 라벨 파일 생성 중...")

# ZIP에서 이미지 추출
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    zip_png_map = {
        Path(f).name: f
        for f in z.namelist()
        if f.endswith(".png") and "train_images" in f
    }

    converted = 0
    skipped   = 0

    for img_id, img_info in images.items():
        fname  = img_info["file_name"]
        width  = img_info["width"]
        height = img_info["height"]

        split  = "train" if img_id in train_img_ids else "val"

        # 이미지 복사
        if fname in zip_png_map:
            img_dst = DATASET_DIR / "images" / split / fname
            if not img_dst.exists():
                with z.open(zip_png_map[fname]) as src:
                    img_dst.write_bytes(src.read())
        else:
            skipped += 1
            continue

        # YOLO txt 생성
        anns = img_anns.get(img_id, [])
        txt_dst = DATASET_DIR / "labels" / split / (Path(fname).stem + ".txt")

        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            yolo_id = coco_id_to_yolo[ann["category_id"]]

            # COCO (x_min, y_min, w, h) → YOLO (cx, cy, w, h) 정규화
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            nw = w / width
            nh = h / height

            # 범위 클램핑 (0~1 벗어나면 보정)
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines.append(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        txt_dst.write_text("\n".join(lines), encoding="utf-8")
        converted += 1

print(f"  변환 완료: {converted}개")
print(f"  스킵 (이미지 없음): {skipped}개")

# pill.yaml 생성 
print("\n[3] pill.yaml 생성 중...")

# 0-based 클래스명 목록 (정렬 순서 유지)
names_list = [id_to_name[i + 1] for i in range(len(id_to_name))]

yaml_content = f"""# 알약 탐지 프로젝트 — 데이터셋 설정
path: {DATASET_DIR.as_posix()}
train: images/train
val:   images/val

nc: {len(names_list)}

names:
"""
for i, name in enumerate(names_list):
    yaml_content += f"  {i}: {name}\n"

yaml_path = OUTPUT_DIR / "pill.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")
print(f"  저장: {yaml_path}")

# 클래스별 split 분포 검증
print("\n[4] Split 분포 검증...")

train_class_count: dict[int, int] = defaultdict(int)
val_class_count:   dict[int, int] = defaultdict(int)

for ann in coco["annotations"]:
    if ann["image_id"] in train_img_ids:
        train_class_count[ann["category_id"]] += 1
    else:
        val_class_count[ann["category_id"]] += 1

# val에 없는 클래스 확인
missing_in_val = [
    id_to_name[cid]
    for cid in train_class_count
    if val_class_count.get(cid, 0) == 0
]

if missing_in_val:
    print(f"  ⚠ val에 없는 클래스 ({len(missing_in_val)}개):")
    for name in missing_in_val:
        print(f"    - {name}")
else:
    print("  모든 클래스가 val에 포함됨")

print("\n" + "=" * 55)
print("변환 완료! 생성된 구조:")
print("=" * 55)
print(f"  {DATASET_DIR}/")
print(f"    images/train/  → {len(list((DATASET_DIR/'images'/'train').glob('*.png')))}장")
print(f"    images/val/    → {len(list((DATASET_DIR/'images'/'val').glob('*.png')))}장")
print(f"    labels/train/  → {len(list((DATASET_DIR/'labels'/'train').glob('*.txt')))}개")
print(f"    labels/val/    → {len(list((DATASET_DIR/'labels'/'val').glob('*.txt')))}개")
print(f"  {yaml_path}")
