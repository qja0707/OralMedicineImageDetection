"""
소수 클래스 Crop & Paste 증강
------------------------------
실행 순서: merge_annotations.py → convert_to_yolo.py → [이 파일] → train_experiment.py

동작 방식:
  1. train_coco.json에서 소수 클래스(N개 이하) 어노테이션 수집
  2. 해당 알약을 bbox 기준으로 crop
  3. train 폴더의 랜덤 배경 이미지에 합성
  4. 새 이미지와 YOLO 라벨을 dataset/images/train, dataset/labels/train에 저장

결과:
  - 소수 클래스 알약이 다양한 배경에서 등장하는 새 학습 이미지 생성
  - 기존 train 폴더에 직접 추가되므로 train_experiment.py 그대로 사용 가능
"""

import json
import random
import shutil
import zipfile
import io
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageEnhance
import numpy as np

# ── 경로 설정 ──────────────────────────────────────────────────
ZIP_PATH     = r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\ai10-level1-project.zip"
OUTPUT_DIR   = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\output")
COCO_JSON    = OUTPUT_DIR / "train_coco.json"
MAPPING_JSON = OUTPUT_DIR / "category_mapping.json"
DATASET_DIR  = OUTPUT_DIR / "dataset"
TRAIN_IMG    = DATASET_DIR / "images" / "train"
TRAIN_LBL    = DATASET_DIR / "labels" / "train"

# ── 설정 ───────────────────────────────────────────────────────
MINORITY_THRESHOLD = 5    # 이 수 이하 어노테이션을 소수 클래스로 판단
N_PER_PILL        = 15   # 알약 1개당 생성할 이미지 수
RANDOM_SEED       = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 데이터 로드 ────────────────────────────────────────────────
print("=" * 60)
print("소수 클래스 Crop & Paste 증강 시작")
print("=" * 60)

with open(COCO_JSON, encoding="utf-8") as f:
    coco = json.load(f)
with open(MAPPING_JSON, encoding="utf-8") as f:
    mapping = json.load(f)

id_to_name  = {int(k): v for k, v in mapping["id_to_name"].items()}
name_to_idx = {v: int(k)-1 for k, v in mapping["id_to_name"].items()}
# category_id → YOLO class index (0-based)
cat_to_idx  = {}
sorted_cats = sorted([int(k) for k in mapping["id_to_name"].keys()])
for i, cat_id in enumerate(sorted_cats):
    cat_to_idx[cat_id] = i

img_info_map = {img["id"]: img for img in coco["images"]}

# ── 클래스별 어노테이션 수 집계 ────────────────────────────────
class_count = defaultdict(int)
for ann in coco["annotations"]:
    class_count[ann["category_id"]] += 1

# ── 소수 클래스 확인 ───────────────────────────────────────────
minority_cats = {
    cat_id: count
    for cat_id, count in class_count.items()
    if count <= MINORITY_THRESHOLD
}

print(f"\n소수 클래스 (어노테이션 {MINORITY_THRESHOLD}개 이하): {len(minority_cats)}종")
for cat_id, count in sorted(minority_cats.items(), key=lambda x: x[1]):
    print(f"  {id_to_name.get(cat_id, str(cat_id))[:30]:<30} : {count}개")

# ── 소수 클래스 어노테이션 수집 ────────────────────────────────
minority_anns = [
    ann for ann in coco["annotations"]
    if ann["category_id"] in minority_cats
]
print(f"\n소수 클래스 어노테이션 총 {len(minority_anns)}개")

# ── 배경 이미지 목록 (train 폴더 전체) ────────────────────────
bg_paths = list(TRAIN_IMG.glob("*.png"))
print(f"배경 이미지 풀: {len(bg_paths)}장")
print(f"알약 1개당 생성: {N_PER_PILL}장")
print(f"예상 생성 이미지: {len(minority_anns) * N_PER_PILL}장\n")

# ── ZIP에서 이미지 로드 함수 ───────────────────────────────────
def load_image_from_zip(z, fname, png_map):
    if fname not in png_map:
        return None
    with z.open(png_map[fname]) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


# ── 증강 함수 (crop된 알약에 적용) ────────────────────────────
def augment_pill(pill_img: Image.Image) -> Image.Image:
    """crop된 알약에 색상/밝기 변환 적용"""
    # 밝기 변환
    brightness = ImageEnhance.Brightness(pill_img)
    pill_img = brightness.enhance(random.uniform(0.7, 1.3))

    # 채도 변환
    color = ImageEnhance.Color(pill_img)
    pill_img = color.enhance(random.uniform(0.8, 1.2))

    # 대비 변환
    contrast = ImageEnhance.Contrast(pill_img)
    pill_img = contrast.enhance(random.uniform(0.8, 1.2))

    # 90도 단위 회전만 (bbox 오염 방지)
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        pill_img = pill_img.rotate(angle, expand=True)

    # 좌우 반전
    if random.random() > 0.5:
        pill_img = pill_img.transpose(Image.FLIP_LEFT_RIGHT)

    return pill_img


# ── 메인 실행 ──────────────────────────────────────────────────
generated = 0
skipped   = 0

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    png_map = {
        Path(f).name: f
        for f in z.namelist()
        if f.endswith(".png") and "train_images" in f
    }

    for ann in minority_anns:
        img_info = img_info_map.get(ann["image_id"])
        if img_info is None:
            skipped += 1
            continue

        fname  = img_info["file_name"]
        img_w  = img_info["width"]
        img_h  = img_info["height"]

        # 원본 이미지 로드
        src_img = load_image_from_zip(z, fname, png_map)
        if src_img is None:
            skipped += 1
            continue

        # bbox crop (여유 margin 10% 추가)
        x, y, w, h = ann["bbox"]
        margin_x = w * 0.1
        margin_y = h * 0.1
        x1 = max(0, int(x - margin_x))
        y1 = max(0, int(y - margin_y))
        x2 = min(img_w, int(x + w + margin_x))
        y2 = min(img_h, int(y + h + margin_y))

        pill_crop = src_img.crop((x1, y1, x2, y2))
        pill_w    = x2 - x1
        pill_h    = y2 - y1

        if pill_w <= 0 or pill_h <= 0:
            skipped += 1
            continue

        cat_id  = ann["category_id"]
        cat_idx = cat_to_idx[cat_id]

        for i in range(N_PER_PILL):
            # 배경 이미지 랜덤 선택
            bg_path = random.choice(bg_paths)
            bg_img  = Image.open(bg_path).convert("RGB")
            bg_w, bg_h = bg_img.size

            # 알약 증강 적용
            aug_pill = augment_pill(pill_crop.copy())
            aw, ah   = aug_pill.size

            # 이미지보다 알약이 크면 스킵
            if aw >= bg_w or ah >= bg_h:
                continue

            # 랜덤 위치에 붙여넣기
            paste_x = random.randint(0, bg_w - aw)
            paste_y = random.randint(0, bg_h - ah)
            bg_img.paste(aug_pill, (paste_x, paste_y))

            # 저장 이름
            save_stem = f"minority_cat{cat_id}_ann{ann['id']}_{i:03d}"
            bg_img.save(TRAIN_IMG / f"{save_stem}.png")

            # YOLO 라벨 계산
            cx = (paste_x + aw / 2) / bg_w
            cy = (paste_y + ah / 2) / bg_h
            nw = aw / bg_w
            nh = ah / bg_h

            # 범위 클램핑
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))

            with open(TRAIN_LBL / f"{save_stem}.txt", "w") as lf:
                lf.write(f"{cat_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            generated += 1

        print(f"  [{id_to_name.get(cat_id,'?')[:20]:<20}] ann_id={ann['id']} → {N_PER_PILL}장 생성")

# ── 결과 요약 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("소수 클래스 증강 완료")
print("=" * 60)
print(f"  생성된 이미지 수 : {generated}장")
print(f"  스킵된 어노테이션: {skipped}개")

# train 폴더 최종 크기 확인
total_train = len(list(TRAIN_IMG.glob("*.png")))
total_label = len(list(TRAIN_LBL.glob("*.txt")))
print(f"\n  train 이미지 총계: {total_train}장 (기존 186 + 신규 {total_train-186})")
print(f"  train 라벨 총계  : {total_label}개")
print(f"\n  저장 위치:")
print(f"    이미지: {TRAIN_IMG}")
print(f"    라벨  : {TRAIN_LBL}")
print("\n  다음 단계: python train_experiment.py")
print("  (추가된 이미지 포함하여 자동 학습)")
print("=" * 60)
