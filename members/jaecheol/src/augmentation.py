"""
데이터 증강
------------------------------
- Albumentations 기반 공통 증강
- 모델 포맷과 무관하게 동작 (COCO bbox 기준)
- 증강 전/후 시각화 포함

설치: pip install albumentations Pillow
"""

import json
import random
import zipfile
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
random.seed(42)
np.random.seed(42)

# 경로 설정 
ZIP_PATH   = r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\ai10-level1-project.zip"
OUTPUT_DIR = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\output")
COCO_JSON  = OUTPUT_DIR / "train_coco.json"
MAPPING_JSON = OUTPUT_DIR / "category_mapping.json"
AUG_DIR    = OUTPUT_DIR / "augmentation_samples"
AUG_DIR.mkdir(exist_ok=True)

with open(COCO_JSON, encoding="utf-8") as f:
    coco = json.load(f)
with open(MAPPING_JSON, encoding="utf-8") as f:
    mapping = json.load(f)

id_to_name = {int(k): v for k, v in mapping["id_to_name"].items()}
img_anns   = defaultdict(list)
for ann in coco["annotations"]:
    img_anns[ann["image_id"]].append(ann)
images = {img["id"]: img for img in coco["images"]}

# 증강 정의 
def get_train_transform(img_size: int = 640) -> A.Compose:
    """
    학습용 증강 파이프라인
    bbox_params: COCO 포맷 [x_min, y_min, w, h] 절댓값 좌표
    """
    return A.Compose([

        # 1. 기하학적 변환 
        A.RandomRotate90(p=0.5),        # 90도 단위 회전
        A.Rotate(                        # 자유 회전 ±180도
            limit=180,
            border_mode=0,              # 빈 부분 검정으로 채움
            p=0.7
        ),
        A.HorizontalFlip(p=0.5),        # 좌우 반전
        A.VerticalFlip(p=0.5),          # 상하 반전

        # 2. 크기/위치 변환
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.7, 1.0),           # 70~100% 크기로 crop
            ratio=(0.9, 1.1),
            p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.3,
            rotate_limit=0,             # 회전은 위에서 처리
            border_mode=0,
            p=0.4
        ),

        # 3. 색상 변환
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.4,
            hue=0.1,                    # 색조는 약하게 (알약 색상 보존)
            p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=40,
            val_shift_limit=30,
            p=0.4
        ),

        # 4. 노이즈/품질
        A.GaussNoise(
            std_range=(0.01, 0.05),     # 약한 노이즈 (각인 보존)
            p=0.3
        ),
        A.ImageCompression(
            quality_range=(80, 100),    # JPEG 압축 시뮬레이션
            p=0.2
        ),

        # 5. 가리기 (Occlusion)
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(20, 60),
            hole_width_range=(20, 60),
            fill=0,                     # 검정으로 가림
            p=0.3
        ),

        # 6. 최종 리사이즈
        A.Resize(img_size, img_size),

    ],
    bbox_params=A.BboxParams(
        format='coco',                  # [x_min, y_min, w, h] 절댓값
        label_fields=['category_ids'],
        min_visibility=0.3,             # 30% 미만 가려진 bbox 제거
        clip=True,
    ))


def get_val_transform(img_size: int = 640) -> A.Compose:
    """검증/추론용 변환 — 리사이즈만"""
    return A.Compose([
        A.Resize(img_size, img_size),
    ],
    bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
        clip=True,
    ))


# 증강 적용 함수 
def apply_augmentation(
    image: np.ndarray,
    bboxes: list,          # [[x,y,w,h], ...] COCO 절댓값
    category_ids: list,    # [cat_id, ...]
    transform: A.Compose,
) -> tuple:
    """
    증강 적용 후 (image, bboxes, category_ids) 반환
    bboxes: COCO 포맷 유지
    """
    result = transform(
        image        = image,
        bboxes       = bboxes,
        category_ids = category_ids,
    )
    return (
        result["image"],
        list(result["bboxes"]),
        list(result["category_ids"]),
    )


# 증강 시각화 
def visualize_augmentations(image_id: int, n_aug: int = 5):
    """원본 + 증강 n개를 나란히 시각화"""
    img_info = images.get(image_id)
    if img_info is None:
        return

    anns = img_anns.get(image_id, [])
    bboxes      = [ann["bbox"] for ann in anns]
    category_ids= [ann["category_id"] for ann in anns]
    colors_bbox = ['#E24B4A', '#1D9E75', '#378ADD', '#EF9F27']

    # 이미지 로드
    fname = img_info["file_name"]
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        png_map = {Path(f).name: f for f in z.namelist()
                   if f.endswith(".png") and "train_images" in f}
        if fname not in png_map:
            print(f"  이미지 없음: {fname}")
            return
        with z.open(png_map[fname]) as f:
            img_pil = Image.open(io.BytesIO(f.read())).convert("RGB")

    img_np = np.array(img_pil)
    transform = get_train_transform(640)

    fig, axes = plt.subplots(1, n_aug + 1, figsize=(4 * (n_aug + 1), 5))

    def draw_boxes(ax, img, boxes, cat_ids, title):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=8)
        h, w = img.shape[:2]
        for i, (box, cid) in enumerate(zip(boxes, cat_ids)):
            x, y, bw, bh = box
            # 정규화 여부 자동 판별
            if max(x, y, bw, bh) <= 1.0:
                x, y, bw, bh = x*w, y*h, bw*w, bh*h
            color = colors_bbox[i % len(colors_bbox)]
            rect = patches.Rectangle((x, y), bw, bh,
                                      linewidth=1.5, edgecolor=color,
                                      facecolor='none')
            ax.add_patch(rect)
            name = id_to_name.get(cid, str(cid))[:8]
            ax.text(x, max(0, y-3), name, color=color, fontsize=6,
                    bbox=dict(facecolor='white', alpha=0.6,
                              edgecolor='none', pad=1))

    # 원본
    draw_boxes(axes[0], img_np, bboxes, category_ids, "원본")

    # 증강 n개
    for idx in range(n_aug):
        try:
            aug_img, aug_boxes, aug_cats = apply_augmentation(
                img_np, bboxes, category_ids, transform)
            draw_boxes(axes[idx+1], aug_img, aug_boxes, aug_cats,
                       f"증강 {idx+1}")
        except Exception as e:
            axes[idx+1].set_title(f"증강 {idx+1}\n오류")
            axes[idx+1].axis('off')

    plt.suptitle(f"증강 전/후 비교\n{fname[:40]}", fontsize=10)
    plt.tight_layout()
    save_path = AUG_DIR / f"aug_sample_{image_id}.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


# 실행 
print("=" * 60)
print("증강 파이프라인 시각화")
print("=" * 60)

# 알약 4개짜리 이미지 3장 선택해서 시각화
sample_ids = [img["id"] for img in coco["images"]
              if len(img_anns.get(img["id"], [])) == 4][:3]

for img_id in sample_ids:
    print(f"\n  이미지 ID {img_id} 증강 시각화 중...")
    visualize_augmentations(img_id, n_aug=4)

print("\n" + "=" * 60)
print("증강 파이프라인 요약")
print("=" * 60)
print("  적용 증강 목록:")
print("    1. RandomRotate90 + Rotate(±180°) — 알약은 방향 무관")
print("    2. HorizontalFlip + VerticalFlip  — 대칭 알약 다양성")
print("    3. RandomResizedCrop              — 크기/위치 다양성")
print("    4. ColorJitter + HSV              — 배경/조명 다양성")
print("    5. GaussNoise                     — 실제 촬영 노이즈")
print("    6. CoarseDropout                  — 부분 가림 대응")
print(f"\n  시각화 저장: {AUG_DIR}")
print("\n  다음 단계: dataset_builder.py 실행")
print("  (모델별 포맷 변환: YOLO txt / COCO JSON)")
print("=" * 60)

# 모델별 사용 가이드 출력
print("\n" + "=" * 60)
print("모델별 증강 적용 방법")
print("=" * 60)
print("""
  [YOLOv11 / RT-DETRv2] — ultralytics 내장 증강 사용
    model.train(
        mosaic=1.0, degrees=180, flipud=0.5, fliplr=0.5,
        hsv_s=0.7, hsv_v=0.4, copy_paste=0.3
    )
    → pill.yaml만 있으면 됨. Albumentations 별도 불필요.

  [EfficientViT / DINO] — 커스텀 DataLoader에 적용
    transform = get_train_transform(640)
    aug_img, aug_boxes, aug_cats = apply_augmentation(
        image, bboxes, category_ids, transform
    )
    → 이 파일의 함수를 DataLoader의 __getitem__에서 호출.
""")
