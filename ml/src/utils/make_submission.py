"""
Kaggle 제출용 CSV 생성 (실제 category_id 적용 + TTA)
------------------------------------------------
YOLOv11l best.pt로 test 이미지 예측 후
실제 Kaggle category_id로 변환하여 CSV 생성

실행 순서:
  1. python fix_category_mapping.py  (최초 1회만)
  2. python make_submission.py
"""

import json
import csv
import zipfile
from pathlib import Path
from ultralytics import YOLO

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR     = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트")
OUTPUT_DIR   = BASE_DIR / "output"
ZIP_PATH     = BASE_DIR / "ai10-level1-project.zip"

# YOLOv11l 6차 실험 best.pt (최고 성능)
MODEL_PT = OUTPUT_DIR / "runs" / "exp_yolo11l_final" / "weights" / "best.pt"

# test 이미지 폴더
TEST_DIR     = BASE_DIR / "sprint_ai_project1_data" / "test_images"

# 카테고리 매핑
MAPPING_JSON      = OUTPUT_DIR / "category_mapping.json"
REAL_MAPPING_JSON = OUTPUT_DIR / "real_category_mapping.json"

# 출력 파일
OUTPUT_CSV   = BASE_DIR / "submission.csv"

# ── 설정 ───────────────────────────────────────────────────────
CONF_THRESH  = 0.03
IOU_THRESH   = 0.45
IMG_SIZE     = 1280


# ── 실제 category_id 매핑 로드 ────────────────────────────────
# real_category_mapping.json 없으면 자동 생성
if not REAL_MAPPING_JSON.exists():
    print("실제 category_id 매핑 파일 생성 중...")

    with open(MAPPING_JSON, encoding="utf-8") as f:
        our_mapping = json.load(f)

    our_id_to_name  = {int(k): v for k, v in our_mapping["id_to_name"].items()}
    sorted_cats     = sorted(our_id_to_name.keys())
    our_idx_to_name = {i: our_id_to_name[cid] for i, cid in enumerate(sorted_cats)}

    real_cats = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        json_files = [f for f in z.namelist() if f.endswith('.json')]
        for jf_name in json_files:
            with z.open(jf_name) as jf:
                data = json.load(jf)
                if 'categories' in data:
                    for cat in data['categories']:
                        real_cats[cat['id']] = cat['name']

    idx_to_real_cat = {}
    for idx, our_name in our_idx_to_name.items():
        for real_id, real_name in real_cats.items():
            if our_name in real_name or real_name in our_name:
                idx_to_real_cat[idx] = real_id
                break

    output = {
        "idx_to_real_category_id": {str(k): v for k, v in idx_to_real_cat.items()},
        "real_id_to_name"        : {str(k): v for k, v in real_cats.items()},
    }
    with open(REAL_MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료: {REAL_MAPPING_JSON}")

with open(REAL_MAPPING_JSON, encoding="utf-8") as f:
    real_mapping = json.load(f)

idx_to_real_cat = {int(k): v for k, v in real_mapping["idx_to_real_category_id"].items()}

print("=" * 60)
print("Kaggle 제출용 CSV 생성 (실제 category_id 적용 + TTA)")
print("=" * 60)
print(f"  모델 : {MODEL_PT.name}")
print(f"  test : {TEST_DIR}")
print(f"  conf : {CONF_THRESH}")


# ── 모델 로드 ─────────────────────────────────────────────────
model = YOLO(str(MODEL_PT))

# ── test 이미지 목록 ───────────────────────────────────────────
img_extensions = [".jpg", ".jpeg", ".png"]
test_images = sorted(
    [p for p in TEST_DIR.iterdir() if p.suffix.lower() in img_extensions],
    key=lambda p: int("".join(filter(str.isdigit, p.stem)))
)
print(f"  이미지 : {len(test_images)}장\n")


# ── 예측 및 CSV 생성 ───────────────────────────────────────────
rows = []
annotation_id = 1

for img_path in test_images:
    # image_id : 파일명 숫자
    digits   = "".join(filter(str.isdigit, img_path.stem))
    image_id = int(digits) if digits else 0

    results = model(
        str(img_path),
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        imgsz   = IMG_SIZE,
        verbose = False,
        augment = True,   # TTA 적용
    )

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes   = result.boxes.xyxy.cpu().numpy()
        scores  = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_idx in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box
            bbox_x = round(float(x1), 2)
            bbox_y = round(float(y1), 2)
            bbox_w = round(float(x2 - x1), 2)
            bbox_h = round(float(y2 - y1), 2)

            # 실제 Kaggle category_id 사용
            category_id = idx_to_real_cat.get(cls_idx, -1)
            if category_id == -1:
                continue  # 매핑 실패 bbox 제외

            rows.append({
                "annotation_id": annotation_id,
                "image_id"     : image_id,
                "category_id"  : category_id,
                "bbox_x"       : bbox_x,
                "bbox_y"       : bbox_y,
                "bbox_w"       : bbox_w,
                "bbox_h"       : bbox_h,
                "score"        : round(float(score), 4),
            })
            annotation_id += 1

    print(f"  [{image_id:>6}] {img_path.name} → {annotation_id-1}개 누적")


# ── CSV 저장 ───────────────────────────────────────────────────
fieldnames = [
    "annotation_id", "image_id", "category_id",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 60)
print("CSV 생성 완료")
print("=" * 60)
print(f"  총 예측 bbox : {len(rows)}개")
print(f"  저장 위치    : {OUTPUT_CSV}")
print("=" * 60)
