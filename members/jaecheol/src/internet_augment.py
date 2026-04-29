"""
인터넷 이미지 Crop & Paste 증강
---------------------------------
탐지 실패 클래스의 인터넷 이미지를 배경에 합성하여
학습 데이터를 추가 생성하는 코드

대상 클래스:
  - 큐시드정 31.5mg/PTP    (mAP=0)
  - 놀텍정 10mg            (Recall=0)
  - 맥시부펜이알정 300mg    (Recall=0)
  - 아질렉트정             (Recall=0)
  - 에스원엠프정 20mg      (Recall=0)
  - 비모보정 500/20mg      (Recall=0.498)

실행 순서:
  1. 이미지 파일을 INTERNET_IMG_DIR에 저장
  2. python internet_augment.py
  3. python train_experiment.py
"""

import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR     = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트")
OUTPUT_DIR   = BASE_DIR / "output"
MAPPING_JSON = OUTPUT_DIR / "category_mapping.json"
DATASET_DIR  = OUTPUT_DIR / "dataset"
TRAIN_IMG    = DATASET_DIR / "images" / "train"
TRAIN_LBL    = DATASET_DIR / "labels" / "train"

# ── 인터넷 이미지 폴더 ─────────────────────────────────────────
# 이 폴더에 원본 이미지 파일들을 저장해주세요
INTERNET_IMG_DIR = BASE_DIR / "internet_pills"

# ── 설정 ───────────────────────────────────────────────────────
N_PER_IMG   = 15    # 이미지 1장당 생성할 이미지 수
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 인터넷 이미지 → 클래스명 매핑 ────────────────────────────
# 파일명과 클래스명을 정확히 맞춰주세요
IMG_TO_CLASS = {
    "큐시드정_원본.png"     : "큐시드정 31.5mg/PTP",
    "놀텍정_원본.png"       : "놀텍정 10mg",
    "맥시부펜이알정_원본.png": "맥시부펜이알정 300mg",
    "아질렉트정_원본.png"   : "아질렉트정(라사길린메실산염)",
    "에스원엠프정_원본.png"  : "에스원엠프정 20mg",
    "비모보정_원본.png"     : "비모보정 500/20mg",
}

# ── 카테고리 매핑 로드 ─────────────────────────────────────────
with open(MAPPING_JSON, encoding="utf-8") as f:
    mapping = json.load(f)

id_to_name  = {int(k): v for k, v in mapping["id_to_name"].items()}
name_to_id  = {v: int(k) for k, v in mapping["id_to_name"].items()}
sorted_cats = sorted([int(k) for k in mapping["id_to_name"].keys()])
cat_to_idx  = {cat_id: i for i, cat_id in enumerate(sorted_cats)}

# ── 배경 이미지 목록 ───────────────────────────────────────────
bg_paths = [p for p in TRAIN_IMG.glob("*.png") if not p.stem.startswith("minority") and not p.stem.startswith("internet")]
print(f"배경 이미지 풀: {len(bg_paths)}장")


# ── 흰 배경 제거 함수 ─────────────────────────────────────────
def remove_white_background(img: Image.Image, threshold=240) -> Image.Image:
    """흰색/밝은 배경을 투명하게 변환"""
    img = img.convert("RGBA")
    data = np.array(img)

    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]

    # 흰색에 가까운 픽셀을 투명하게
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)
    data[white_mask, 3] = 0

    return Image.fromarray(data, "RGBA")


# ── 알약 자동 crop 함수 ───────────────────────────────────────
def auto_crop_pill(img: Image.Image) -> Image.Image:
    """배경 제거 후 알약 영역만 crop"""
    img_rgba = remove_white_background(img)
    bbox = img_rgba.getbbox()  # 투명하지 않은 영역의 bbox
    if bbox is None:
        return img_rgba
    # margin 추가
    margin = 10
    x1 = max(0, bbox[0] - margin)
    y1 = max(0, bbox[1] - margin)
    x2 = min(img_rgba.width,  bbox[2] + margin)
    y2 = min(img_rgba.height, bbox[3] + margin)
    return img_rgba.crop((x1, y1, x2, y2))


# ── 증강 함수 ─────────────────────────────────────────────────
def augment_pill(pill_img: Image.Image) -> Image.Image:
    """알약에 색상/밝기/회전 변환 적용"""
    pill_rgb = pill_img.convert("RGB")

    brightness = ImageEnhance.Brightness(pill_rgb)
    pill_rgb = brightness.enhance(random.uniform(0.7, 1.3))

    contrast = ImageEnhance.Contrast(pill_rgb)
    pill_rgb = contrast.enhance(random.uniform(0.8, 1.2))

    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        pill_rgb = pill_rgb.rotate(angle, expand=True)

    if random.random() > 0.5:
        pill_rgb = pill_rgb.transpose(Image.FLIP_LEFT_RIGHT)

    return pill_rgb


# ── 메인 실행 ─────────────────────────────────────────────────
print("=" * 60)
print("인터넷 이미지 Crop & Paste 증강 시작")
print("=" * 60)

INTERNET_IMG_DIR.mkdir(exist_ok=True)

generated = 0
skipped   = 0

for fname, class_name in IMG_TO_CLASS.items():
    img_path = INTERNET_IMG_DIR / fname

    # 파일 없으면 스킵
    if not img_path.exists():
        print(f"  ⚠️  파일 없음: {fname} → 스킵")
        skipped += 1
        continue

    # 클래스명으로 category_id 찾기
    cat_id = None
    for cid, cname in id_to_name.items():
        if class_name in cname or cname in class_name:
            cat_id = cid
            break

    if cat_id is None:
        print(f"  ⚠️  클래스 매핑 실패: {class_name} → 스킵")
        skipped += 1
        continue

    cat_idx = cat_to_idx[cat_id]

    # 이미지 로드 및 배경 제거
    src_img  = Image.open(img_path)
    pill_crop = auto_crop_pill(src_img)
    pill_w, pill_h = pill_crop.size

    print(f"\n  [{class_name[:25]:<25}] cat_id={cat_id} → {N_PER_IMG}장 생성")

    for i in range(N_PER_IMG):
        # 배경 이미지 랜덤 선택
        bg_path = random.choice(bg_paths)
        bg_img  = Image.open(bg_path).convert("RGB")
        bg_w, bg_h = bg_img.size

        # 알약 증강
        aug_pill = augment_pill(pill_crop.copy())
        aw, ah   = aug_pill.size

        # 크기 조정 (배경의 10~25% 크기로)
        scale    = random.uniform(0.10, 0.25)
        new_w    = int(bg_w * scale)
        new_h    = int(new_w * ah / aw)
        aug_pill = aug_pill.resize((new_w, new_h), Image.LANCZOS)
        aw, ah   = aug_pill.size

        if aw >= bg_w or ah >= bg_h:
            continue

        # 랜덤 위치에 붙여넣기
        paste_x = random.randint(0, bg_w - aw)
        paste_y = random.randint(0, bg_h - ah)
        bg_img.paste(aug_pill, (paste_x, paste_y))

        # 저장
        save_stem = f"internet_cat{cat_id}_{fname.replace('.png','')}_{i:03d}"
        bg_img.save(TRAIN_IMG / f"{save_stem}.png")

        # YOLO 라벨
        cx = (paste_x + aw / 2) / bg_w
        cy = (paste_y + ah / 2) / bg_h
        nw = aw / bg_w
        nh = ah / bg_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.001, min(1.0, nw))
        nh = max(0.001, min(1.0, nh))

        with open(TRAIN_LBL / f"{save_stem}.txt", "w") as lf:
            lf.write(f"{cat_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        generated += 1

# ── 결과 요약 ─────────────────────────────────────────────────
total_train = len(list(TRAIN_IMG.glob("*.png")))
print("\n" + "=" * 60)
print("인터넷 이미지 증강 완료")
print("=" * 60)
print(f"  생성된 이미지 수  : {generated}장")
print(f"  스킵된 이미지     : {skipped}개")
print(f"  train 이미지 총계 : {total_train}장")
print(f"\n  저장 위치:")
print(f"    이미지: {TRAIN_IMG}")
print(f"    라벨  : {TRAIN_LBL}")
print("\n  다음 단계: python train_experiment.py")
print("=" * 60)
