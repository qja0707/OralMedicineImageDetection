"""
Pseudo Labeling 파이프라인
---------------------------
1. submission.csv에서 고신뢰도 예측 추출
2. test 이미지 + 라벨을 train에 추가
3. YOLOv11l 재학습

실행:
  python pseudo_labeling.py
"""

import csv
import shutil
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR     = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트")
OUTPUT_DIR   = BASE_DIR / "output"
TEST_DIR     = BASE_DIR / "sprint_ai_project1_data" / "test_images"
TRAIN_IMG    = OUTPUT_DIR / "dataset" / "images" / "train"
TRAIN_LBL    = OUTPUT_DIR / "dataset" / "labels" / "train"
SUBMISSION   = BASE_DIR / "submission.csv"

# ── 설정 ───────────────────────────────────────────────────────
CONF_THRESH  = 0.7   # 이 이상 score만 라벨로 사용
IMG_SIZE     = 1280  # 원본 이미지 크기 (정규화에 사용)

# ── category_id → YOLO class index 매핑 ───────────────────────
import json
REAL_MAPPING = OUTPUT_DIR / "real_category_mapping.json"
MAPPING_JSON = OUTPUT_DIR / "category_mapping.json"

with open(REAL_MAPPING, encoding="utf-8") as f:
    real_mapping = json.load(f)
with open(MAPPING_JSON, encoding="utf-8") as f:
    our_mapping = json.load(f)

# real category_id → YOLO class index
idx_to_real = {int(k): v for k, v in real_mapping["idx_to_real_category_id"].items()}
real_to_idx = {v: int(k) for k, v in idx_to_real.items()}

our_id_to_name = {int(k): v for k, v in our_mapping["id_to_name"].items()}
sorted_cats    = sorted(our_id_to_name.keys())
cat_to_idx     = {cat_id: i for i, cat_id in enumerate(sorted_cats)}

# real category_id → YOLO class index
real_cat_to_yolo_idx = {}
for real_cat_id, yolo_idx in [(v, int(k)) for k, v in idx_to_real.items()]:
    real_cat_to_yolo_idx[real_cat_id] = cat_to_idx.get(yolo_idx, yolo_idx)

print("=" * 60)
print("Pseudo Labeling 파이프라인")
print("=" * 60)
print(f"  submission : {SUBMISSION}")
print(f"  conf 기준  : score >= {CONF_THRESH}")

# ── submission.csv 로드 ───────────────────────────────────────
rows_by_image = {}  # {image_id: [row, ...]}

with open(SUBMISSION, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        score = float(row["score"])
        if score < CONF_THRESH:
            continue
        image_id = int(row["image_id"])
        if image_id not in rows_by_image:
            rows_by_image[image_id] = []
        rows_by_image[image_id].append(row)

print(f"\n  고신뢰도 이미지 수 : {len(rows_by_image)}장")
total_boxes = sum(len(v) for v in rows_by_image.values())
print(f"  고신뢰도 bbox 수  : {total_boxes}개")

# ── test 이미지 목록 ───────────────────────────────────────────
img_extensions = [".jpg", ".jpeg", ".png"]
test_images = {
    int("".join(filter(str.isdigit, p.stem))): p
    for p in TEST_DIR.iterdir()
    if p.suffix.lower() in img_extensions
}

# ── test 이미지 → train 복사 + YOLO 라벨 생성 ─────────────────
print(f"\n  train에 추가 중...")
added = 0
skipped = 0

for image_id, boxes in rows_by_image.items():
    img_path = test_images.get(image_id)
    if img_path is None:
        skipped += 1
        continue

    # 이미지 복사
    save_stem = f"pseudo_{image_id}"
    save_img  = TRAIN_IMG / f"{save_stem}{img_path.suffix}"
    save_lbl  = TRAIN_LBL / f"{save_stem}.txt"

    if save_img.exists():
        skipped += 1
        continue

    shutil.copy2(img_path, save_img)

    # 실제 이미지 크기 확인
    from PIL import Image as PILImage
    with PILImage.open(img_path) as img:
        img_w, img_h = img.size

    # YOLO 라벨 생성
    with open(save_lbl, "w") as lf:
        for row in boxes:
            cat_id  = int(row["category_id"])
            bbox_x  = float(row["bbox_x"])
            bbox_y  = float(row["bbox_y"])
            bbox_w  = float(row["bbox_w"])
            bbox_h  = float(row["bbox_h"])

            # bbox → YOLO 정규화
            cx = (bbox_x + bbox_w / 2) / img_w
            cy = (bbox_y + bbox_h / 2) / img_h
            nw = bbox_w / img_w
            nh = bbox_h / img_h

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))

            # real category_id → YOLO class index
            yolo_idx = real_cat_to_yolo_idx.get(cat_id, -1)
            if yolo_idx == -1:
                continue

            lf.write(f"{yolo_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    added += 1

# 현재 train 총계
total_train = len(list(TRAIN_IMG.glob("*.jpg"))) + \
              len(list(TRAIN_IMG.glob("*.png")))

print(f"  추가 완료 : {added}장")
print(f"  스킵      : {skipped}장")
print(f"  train 총계: {total_train}장")
print("\n  다음 단계: python yolo11l_retrain.py")
print("=" * 60)
