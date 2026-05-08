"""
인터넷 이미지 Crop & Paste 증강 (파란 배경 합성 버전 v2)
----------------------------------------------------------
탐지 실패 클래스의 인터넷 이미지를 실제 데이터셋과
동일한 파란 배경에 합성하여 도메인 차이를 최소화

변경사항 (v2):
  - 흰 배경 제거 후 실제 train 이미지(파란 배경)에 RGBA paste
    기존: 흰 배경 그대로 RGB paste → 도메인 불일치
    개선: 알약만 투명 crop 후 파란 배경에 합성 → 도메인 통일
  - N_PER_IMG : 15 → 30

대상 클래스:
  - 큐시드정 31.5mg/PTP  (mAP=0)
  - 놀텍정 10mg          (Recall=0)
  - 맥시부펜이알정 300mg  (Recall=0)
  - 아질렉트정           (Recall=0)
  - 에스원엠프정 20mg    (Recall=0)
  - 비모보정 500/20mg    (Recall=0.498)

실행 순서:
  1. 기존 internet 이미지 삭제
     Get-ChildItem "output\dataset\images\train\internet_*" | Remove-Item
     Get-ChildItem "output\dataset\labels\train\internet_*" | Remove-Item
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
INTERNET_IMG_DIR = BASE_DIR / "internet_pills"

# ── 설정 ───────────────────────────────────────────────────────
N_PER_IMG   = 30    # 15 → 30으로 증가
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 인터넷 이미지 → 클래스명 매핑 ────────────────────────────
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
sorted_cats = sorted([int(k) for k in mapping["id_to_name"].keys()])
cat_to_idx  = {cat_id: i for i, cat_id in enumerate(sorted_cats)}

# ── 배경 이미지 목록 (원본 train 이미지만, 파란 배경) ──────────
bg_paths = [
    p for p in TRAIN_IMG.glob("*.png")
    if not p.stem.startswith("minority") and not p.stem.startswith("internet")
]
print(f"배경 이미지 풀: {len(bg_paths)}장 (파란 배경 원본)")


# ── 흰 배경 제거 → RGBA 알약 crop ─────────────────────────────
def extract_pill_rgba(img: Image.Image, threshold: int = 230) -> Image.Image:
    """
    흰색/밝은 배경을 투명(alpha=0)으로 변환 후
    알약 영역만 타이트하게 crop하여 RGBA 이미지 반환
    """
    rgba = img.convert("RGBA")
    data = np.array(rgba, dtype=np.uint8)
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    # 밝은 배경 마스크 (격자 무늬, 흰색 모두 포함)
    bright_mask = (r > threshold) & (g > threshold) & (b > threshold)
    data[bright_mask, 3] = 0  # 투명 처리

    result = Image.fromarray(data, "RGBA")

    # 알약 영역 bbox로 tight crop
    bbox = result.getbbox()
    if bbox is None:
        return result

    margin = 8
    x1 = max(0, bbox[0] - margin)
    y1 = max(0, bbox[1] - margin)
    x2 = min(result.width,  bbox[2] + margin)
    y2 = min(result.height, bbox[3] + margin)
    return result.crop((x1, y1, x2, y2))


# ── 알약 증강 (RGBA 유지) ──────────────────────────────────────
def augment_pill_rgba(pill: Image.Image) -> Image.Image:
    """
    RGBA 알약 이미지에 밝기/대비/회전/반전 적용
    투명도(alpha) 채널은 보존
    """
    r, g, b, a = pill.split()
    rgb = Image.merge("RGB", (r, g, b))

    rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.75, 1.25))
    rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.85, 1.15))

    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        rgb = rgb.rotate(angle, expand=True)
        a   = a.rotate(angle, expand=True)

    if random.random() > 0.5:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        a   = a.transpose(Image.FLIP_LEFT_RIGHT)

    r2, g2, b2 = rgb.split()
    return Image.merge("RGBA", (r2, g2, b2, a))


# ── 메인 실행 ─────────────────────────────────────────────────
print("=" * 60)
print("인터넷 이미지 Crop & Paste 증강 시작 (파란 배경 합성 v2)")
print("=" * 60)

INTERNET_IMG_DIR.mkdir(exist_ok=True)

generated = 0
skipped   = 0

for fname, class_name in IMG_TO_CLASS.items():
    img_path = INTERNET_IMG_DIR / fname

    if not img_path.exists():
        print(f"  ⚠️  파일 없음: {fname} → 스킵")
        skipped += 1
        continue

    # 클래스 ID 찾기
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

    # 인터넷 이미지에서 알약 RGBA crop
    src_img   = Image.open(img_path)
    pill_rgba = extract_pill_rgba(src_img)

    print(f"\n  [{class_name[:25]:<25}] cat_id={cat_id} → {N_PER_IMG}장 생성")

    for i in range(N_PER_IMG):
        # 파란 배경 이미지 랜덤 선택
        bg_path = random.choice(bg_paths)
        bg_img  = Image.open(bg_path).convert("RGBA")
        bg_w, bg_h = bg_img.size

        # 알약 증강 (RGBA 유지)
        aug_pill = augment_pill_rgba(pill_rgba.copy())
        aw, ah   = aug_pill.size

        # 크기 조정 (배경의 10~22% 크기로)
        scale = random.uniform(0.10, 0.22)
        new_w = int(bg_w * scale)
        new_h = int(new_w * ah / max(aw, 1))
        aug_pill = aug_pill.resize((new_w, new_h), Image.LANCZOS)
        aw, ah   = aug_pill.size

        if aw >= bg_w or ah >= bg_h:
            continue

        # 랜덤 위치에 RGBA paste (투명도 반영)
        paste_x = random.randint(0, bg_w - aw)
        paste_y = random.randint(0, bg_h - ah)
        bg_img.paste(aug_pill, (paste_x, paste_y), aug_pill)  # mask=aug_pill

        # RGB로 변환 후 저장
        final_img = bg_img.convert("RGB")
        save_stem = f"internet_cat{cat_id}_{fname.replace('.png', '')}_{i:03d}"
        final_img.save(TRAIN_IMG / f"{save_stem}.png")

        # YOLO 라벨 생성
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
print("인터넷 이미지 증강 완료 (파란 배경 합성 v2)")
print("=" * 60)
print(f"  생성된 이미지 수  : {generated}장")
print(f"  스킵된 이미지     : {skipped}개")
print(f"  train 이미지 총계 : {total_train}장")
print(f"\n  저장 위치:")
print(f"    이미지: {TRAIN_IMG}")
print(f"    라벨  : {TRAIN_LBL}")
print("\n  다음 단계: python train_experiment.py")
print("=" * 60)
