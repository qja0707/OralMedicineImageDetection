"""
모델별 데이터셋 빌더
-----------------------------
공통 COCO JSON → 각 모델 포맷으로 변환

지원 모델:
  - YOLOv11     : YOLO txt + pill.yaml  (이미 완성, 재사용)
  - RT-DETRv2   : YOLO txt + pill.yaml  (YOLOv11과 동일)
  - EfficientViT: COCO JSON split 버전  (train/val 분리)
  - DINO        : COCO JSON + mmdet config 템플릿

실행: python dataset_builder.py --model all
     python dataset_builder.py --model efficientvit
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

# 경로 설정 
OUTPUT_DIR   = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\output")
COCO_JSON    = OUTPUT_DIR / "train_coco.json"
MAPPING_JSON = OUTPUT_DIR / "category_mapping.json"
DATASET_DIR  = OUTPUT_DIR / "dataset"      # YOLO 포맷 (이미 존재)

with open(COCO_JSON, encoding="utf-8") as f:
    coco = json.load(f)
with open(MAPPING_JSON, encoding="utf-8") as f:
    mapping = json.load(f)

id_to_name = {int(k): v for k, v in mapping["id_to_name"].items()}

# train/val split 이미지 ID 읽기 (convert_to_yolo.py 결과)
train_img_fnames = set(
    p.stem for p in (DATASET_DIR / "images" / "train").glob("*.png")
)
val_img_fnames = set(
    p.stem for p in (DATASET_DIR / "images" / "val").glob("*.png")
)
fname_to_split = {}
for img in coco["images"]:
    stem = Path(img["file_name"]).stem
    if stem in train_img_fnames:
        fname_to_split[img["id"]] = "train"
    elif stem in val_img_fnames:
        fname_to_split[img["id"]] = "val"


# ══════════════════════════════════════════════════════════════
# 빌더 1: YOLOv11 / RT-DETRv2
# ══════════════════════════════════════════════════════════════
def build_yolo(model_name: str = "yolov11"):
    """
    YOLOv11, RT-DETRv2 모두 ultralytics 기반 → 동일 포맷
    
    """
    print(f"\n[{model_name.upper()}] YOLO 포맷 확인...")
    yaml_path = OUTPUT_DIR / "pill.yaml"

    if yaml_path.exists():
        print(f"  pill.yaml 존재: {yaml_path}")
        print(f"  dataset/images/train: "
              f"{len(list((DATASET_DIR/'images'/'train').glob('*.png')))}장")
        print(f"  dataset/images/val  : "
              f"{len(list((DATASET_DIR/'images'/'val').glob('*.png')))}장")
        print(f"\n  학습 코드:")
        if "rtdetr" in model_name.lower():
            print(f"    from ultralytics import RTDETR")
            print(f"    model = RTDETR('rtdetr-l.pt')  # 또는 rtdetr-x.pt")
        else:
            print(f"    from ultralytics import YOLO")
            print(f"    model = YOLO('yolo11m.pt')  # 또는 yolo11l.pt")
        print(f"    model.train(data=r'{yaml_path}', epochs=100, imgsz=640)")
    else:
        print(f"  ⚠ pill.yaml 없음. convert_to_yolo.py 먼저 실행하세요.")


# ══════════════════════════════════════════════════════════════
# 빌더 2: EfficientViT — COCO JSON train/val 분리
# ══════════════════════════════════════════════════════════════
def build_efficientvit():
    """
    EfficientViT는 COCO JSON 포맷 사용
    train_coco.json을 train/val로 분리해서 저장
    """
    print("\n[EfficientViT] COCO JSON split 생성...")
    effvit_dir = OUTPUT_DIR / "efficientvit_data"
    effvit_dir.mkdir(exist_ok=True)

    train_images, val_images = [], []
    train_anns,   val_anns   = [], []

    for img in coco["images"]:
        split = fname_to_split.get(img["id"])
        if split == "train":
            train_images.append(img)
        elif split == "val":
            val_images.append(img)

    train_ids = {img["id"] for img in train_images}
    val_ids   = {img["id"] for img in val_images}

    for ann in coco["annotations"]:
        if ann["image_id"] in train_ids:
            train_anns.append(ann)
        elif ann["image_id"] in val_ids:
            val_anns.append(ann)

    base = {
        "info"      : coco.get("info", {}),
        "categories": coco["categories"],
    }

    train_coco = {**base, "images": train_images, "annotations": train_anns}
    val_coco   = {**base, "images": val_images,   "annotations": val_anns}

    train_path = effvit_dir / "train_coco.json"
    val_path   = effvit_dir / "val_coco.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)

    print(f"  train_coco.json: {len(train_images)}장, {len(train_anns)}개 annotation")
    print(f"  val_coco.json  : {len(val_images)}장,   {len(val_anns)}개 annotation")
    print(f"  저장 위치: {effvit_dir}")

    # EfficientViT DataLoader 템플릿
    template = f'''"""
EfficientViT 커스텀 DataLoader 템플릿
--------------------------------------
설치: pip install git+https://github.com/mit-han-lab/efficientvit
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("efficientvit")  # clone 후 경로

TRAIN_JSON  = r"{train_path}"
VAL_JSON    = r"{val_path}"
IMAGE_DIR   = r"{DATASET_DIR / 'images'}"

class PillDataset(Dataset):
    def __init__(self, json_path, split="train", transform=None):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        self.images     = {{img["id"]: img for img in data["images"]}}
        self.categories = {{cat["id"]: cat for cat in data["categories"]}}
        self.img_anns   = {{}}
        for ann in data["annotations"]:
            self.img_anns.setdefault(ann["image_id"], []).append(ann)
        self.img_ids    = list(self.images.keys())
        self.split      = split
        self.transform  = transform
        self.image_dir  = Path(IMAGE_DIR) / split

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.image_dir / img_info["file_name"]

        image = np.array(Image.open(img_path).convert("RGB"))
        anns  = self.img_anns.get(img_id, [])

        bboxes       = [ann["bbox"] for ann in anns]       # [x,y,w,h]
        category_ids = [ann["category_id"] for ann in anns]

        if self.transform:
            result       = self.transform(image=image, bboxes=bboxes,
                                          category_ids=category_ids)
            image        = result["image"]
            bboxes       = list(result["bboxes"])
            category_ids = list(result["category_ids"])

        return image, bboxes, category_ids

# 사용 예시
# from augmentation import get_train_transform, get_val_transform
# train_ds = PillDataset(TRAIN_JSON, "train", get_train_transform(640))
# val_ds   = PillDataset(VAL_JSON,   "val",   get_val_transform(640))
'''
    template_path = effvit_dir / "pill_dataset.py"
    template_path.write_text(template, encoding="utf-8")
    print(f"  DataLoader 템플릿: {template_path}")


# ══════════════════════════════════════════════════════════════
# 빌더 3: DINO — COCO JSON + mmdetection config 템플릿
# ══════════════════════════════════════════════════════════════
def build_dino():
    """
    DINO는 mmdetection 기반
    COCO JSON은 EfficientViT와 동일하게 재활용
    mmdetection config 파일만 추가 생성
    """
    print("\n[DINO] mmdetection config 템플릿 생성...")
    dino_dir = OUTPUT_DIR / "dino_data"
    dino_dir.mkdir(exist_ok=True)

    # EfficientViT용 split JSON 재활용
    effvit_dir = OUTPUT_DIR / "efficientvit_data"
    if not (effvit_dir / "train_coco.json").exists():
        print("  EfficientViT split JSON 없음 → 먼저 생성...")
        build_efficientvit()

    # mmdetection config 템플릿
    config = f'''
# DINO mmdetection config — 알약 탐지
# 설치: pip install mmdet mmcv-full

_base_ = [
    'configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
]

# 데이터셋 설정
data_root = r"{OUTPUT_DIR}"
num_classes = {len(coco["categories"])}

model = dict(
    bbox_head=dict(num_classes=num_classes)
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=r"{effvit_dir / 'train_coco.json'}",
        data_prefix=dict(img=r"{DATASET_DIR / 'images' / 'train'}"),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=r"{effvit_dir / 'val_coco.json'}",
        data_prefix=dict(img=r"{DATASET_DIR / 'images' / 'val'}"),
    )
)

val_evaluator = dict(
    ann_file=r"{effvit_dir / 'val_coco.json'}"
)

# 학습 설정
train_cfg = dict(max_epochs=50)
optim_wrapper = dict(optimizer=dict(lr=1e-4))
'''
    config_path = dino_dir / "dino_pill.py"
    config_path.write_text(config, encoding="utf-8")
    print(f"  mmdet config: {config_path}")
    print(f"\n  DINO 학습 명령어:")
    print(f"  python tools/train.py {config_path}")
    print(f"\n  ⚠ DINO는 GPU 메모리 20GB+ 필요")
    print(f"  ⚠ mmdetection 별도 설치 필요:")
    print(f"    pip install mmdet mmcv-full")


# ══════════════════════════════════════════════════════════════
# 빌더 4: YOLOv11 + SAM2
# ══════════════════════════════════════════════════════════════
def build_yolo_sam2():
    """
    YOLOv11 탐지 → SAM2 세그멘테이션 → bbox 정밀화 파이프라인
    학습: YOLOv11만 (SAM2는 추론 시 사용)
    """
    print("\n[YOLOv11+SAM2] 파이프라인 안내...")

    template = f'''"""
YOLOv11 + SAM2 파이프라인
--------------------------
학습: YOLOv11 (일반 YOLO 학습과 동일)
추론: YOLOv11 bbox → SAM2로 정밀 마스크 → bbox 재추출

설치:
  pip install ultralytics
  pip install git+https://github.com/facebookresearch/segment-anything-2
"""
from ultralytics import YOLO
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np

# ── 1단계: YOLOv11 학습 (일반 YOLO와 동일) ──────────────────
YAML = r"{OUTPUT_DIR / 'pill.yaml'}"

model_yolo = YOLO("yolo11m.pt")
# model_yolo.train(data=YAML, epochs=100, imgsz=640)

# ── 2단계: SAM2 로드 ─────────────────────────────────────────
sam2_checkpoint = "sam2_hiera_large.pt"
model_cfg       = "sam2_hiera_l.yaml"

predictor = SAM2ImagePredictor(
    build_sam2(model_cfg, sam2_checkpoint)
)

# ── 3단계: 추론 파이프라인 ────────────────────────────────────
def predict_with_sam2(image_path: str, yolo_model, sam_predictor):
    """
    YOLOv11 bbox → SAM2 마스크 → 정밀 bbox 반환
    """
    image = np.array(Image.open(image_path).convert("RGB"))

    # YOLO 탐지
    yolo_results = yolo_model.predict(image_path, conf=0.25, verbose=False)
    boxes = yolo_results[0].boxes

    if boxes is None or len(boxes) == 0:
        return [], [], []

    # SAM2로 각 bbox 마스크 생성
    sam_predictor.set_image(image)
    final_boxes, final_scores, final_classes = [], [], []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        input_box = np.array([x1, y1, x2, y2])

        masks, scores, _ = sam_predictor.predict(
            box           = input_box,
            multimask_output = False,
        )

        # 마스크 → 정밀 bbox 추출
        mask = masks[0]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        # 마스크 기반 tight bbox
        new_x1, new_y1 = float(xs.min()), float(ys.min())
        new_x2, new_y2 = float(xs.max()), float(ys.max())

        final_boxes.append([new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1])
        final_scores.append(float(box.conf[0]))
        final_classes.append(int(box.cls[0]))

    return final_boxes, final_scores, final_classes

# 사용 예시:
# boxes, scores, classes = predict_with_sam2("test.png", model_yolo, predictor)
'''
    sam_dir = OUTPUT_DIR / "yolo_sam2"
    sam_dir.mkdir(exist_ok=True)
    template_path = sam_dir / "pipeline.py"
    template_path.write_text(template, encoding="utf-8")
    print(f"  파이프라인 코드: {template_path}")
    print(f"\n  ⚠ SAM2 설치 필요:")
    print(f"    pip install git+https://github.com/facebookresearch/segment-anything-2")
    print(f"  ⚠ SAM2 체크포인트 다운로드 필요 (~2GB)")


# ── CLI 실행 ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
        choices=["all", "yolov11", "rtdetr", "efficientvit", "dino", "sam2"])
    args = parser.parse_args()

    print("=" * 60)
    print("모델별 데이터셋 빌더")
    print("=" * 60)

    if args.model in ("all", "yolov11"):
        build_yolo("yolov11")
    if args.model in ("all", "rtdetr"):
        build_yolo("rtdetr")
    if args.model in ("all", "efficientvit"):
        build_efficientvit()
    if args.model in ("all", "dino"):
        build_dino()
    if args.model in ("all", "sam2"):
        build_yolo_sam2()

    print("\n" + "=" * 60)
    print("빌드 완료!")
    print("=" * 60)
    print("""
  다음 단계별 학습 명령어:

  [YOLOv11]
    python -c "from ultralytics import YOLO; YOLO('yolo11m.pt').train(data='pill.yaml', epochs=100)"

  [RT-DETRv2]
    python -c "from ultralytics import RTDETR; RTDETR('rtdetr-l.pt').train(data='pill.yaml', epochs=100)"

  [EfficientViT]
    → efficientvit_data/pill_dataset.py 참조

  [DINO]
    → dino_data/dino_pill.py 참조

  [YOLOv11+SAM2]
    → yolo_sam2/pipeline.py 참조
    """)
