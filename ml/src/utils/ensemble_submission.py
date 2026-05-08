"""
YOLOv11l + RT-DETRv2 + RF-DETRv3 3모델 앙상블 제출
-----------------------------------------------------
WBF (Weighted Boxes Fusion) 방식으로
세 모델 예측을 합산하여 최종 CSV 생성

사전 설치:
  pip install ensemble-boxes

실행:
  python ensemble_submission.py
"""

import json
import csv
from pathlib import Path
from ultralytics import YOLO, RTDETR
from PIL import Image

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    print("❌ ensemble-boxes 패키지가 없습니다.")
    print("   pip install ensemble-boxes")
    exit(1)

try:
    from rfdetr import RFDETRLarge
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("⚠️  RF-DETRv3 패키지 없음 → 2모델 앙상블로 진행")

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR   = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트")
OUTPUT_DIR = BASE_DIR / "output"

YOLO_PT    = OUTPUT_DIR / "runs" / "exp_yolo11l_final" / "weights" / "best.pt"
RTDETR_PT = OUTPUT_DIR / "runs" / "exp_rtdetr_retrain" / "weights" / "best_1206.pt"
RFDETR_PT = OUTPUT_DIR / "runs" / "exp_rfdetr_v3_retrain" / "checkpoint_best_regular.pth"

TEST_DIR   = BASE_DIR / "sprint_ai_project1_data" / "test_images"
REAL_MAPPING_JSON = OUTPUT_DIR / "real_category_mapping.json"
OUTPUT_CSV = BASE_DIR / "submission.csv"

# ── 설정 ───────────────────────────────────────────────────────
YOLO_CONF    = 0.01
RTDETR_CONF  = 0.01
RFDETR_CONF  = 0.01
IOU_THRESH   = 0.45
IMG_SIZE     = 1280

WBF_IOU_THR  = 0.6
WBF_SKIP_BOX = 0.01
YOLO_WEIGHT   = 1.0
RTDETR_WEIGHT = 1.0
RFDETR_WEIGHT = 1.0


# ── 카테고리 매핑 로드 ─────────────────────────────────────────
with open(REAL_MAPPING_JSON, encoding="utf-8") as f:
    real_mapping = json.load(f)
idx_to_real_cat = {int(k): v for k, v in real_mapping["idx_to_real_category_id"].items()}

print("=" * 60)
print("3모델 앙상블 제출 (YOLOv11l + RT-DETRv2 + RF-DETRv3)")
print("=" * 60)

# ── 모델 로드 ─────────────────────────────────────────────────
print("  모델 로드 중...")
yolo_model   = YOLO(str(YOLO_PT))
rtdetr_model = RTDETR(str(RTDETR_PT))

rfdetr_model = None
if RFDETR_AVAILABLE:
    try:
        rfdetr_model = RFDETRLarge(pretrain_weights=str(RFDETR_PT))
        print("  RF-DETRv3 로드 완료 ✅")
    except Exception as e:
        print(f"  RF-DETRv3 로드 실패: {e}")
        # .ckpt 파일로 재시도
        try:
            rfdetr_model = RFDETRLarge()
            print("  RF-DETRv3 기본 가중치로 로드 완료 ✅")
        except Exception as e2:
            print(f"  RF-DETRv3 완전 실패: {e2} → 2모델로 진행")

# ── test 이미지 목록 ───────────────────────────────────────────
img_extensions = [".jpg", ".jpeg", ".png"]
test_images = sorted(
    [p for p in TEST_DIR.iterdir() if p.suffix.lower() in img_extensions],
    key=lambda p: int("".join(filter(str.isdigit, p.stem)))
)
print(f"  이미지 : {len(test_images)}장")
print(f"  RF-DETR: {'✅ 사용' if rfdetr_model else '❌ 미사용'}\n")


def extract_boxes_ultralytics(result, img_w, img_h):
    boxes_list, scores_list, labels_list = [], [], []
    if result.boxes is not None and len(result.boxes) > 0:
        for box, score, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int)
        ):
            x1, y1, x2, y2 = box
            boxes_list.append([
                max(0.0, x1/img_w), max(0.0, y1/img_h),
                min(1.0, x2/img_w), min(1.0, y2/img_h)
            ])
            scores_list.append(float(score))
            labels_list.append(int(cls))
    return boxes_list, scores_list, labels_list


def extract_boxes_rfdetr(detections, img_w, img_h):
    boxes_list, scores_list, labels_list = [], [], []
    if detections is None:
        return boxes_list, scores_list, labels_list
    try:
        for box, score, cls in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id
        ):
            if float(score) < RFDETR_CONF:
                continue
            x1, y1, x2, y2 = box
            boxes_list.append([
                max(0.0, float(x1)/img_w), max(0.0, float(y1)/img_h),
                min(1.0, float(x2)/img_w), min(1.0, float(y2)/img_h)
            ])
            scores_list.append(float(score))
            labels_list.append(int(cls))
    except Exception:
        pass
    return boxes_list, scores_list, labels_list


# ── 예측 및 WBF ───────────────────────────────────────────────
rows = []
annotation_id = 1

for img_path in test_images:
    digits   = "".join(filter(str.isdigit, img_path.stem))
    image_id = int(digits) if digits else 0

    # YOLO (TTA)
    yolo_res = yolo_model(
        str(img_path), conf=YOLO_CONF, iou=IOU_THRESH,
        imgsz=IMG_SIZE, verbose=False, augment=True,
    )[0]

    # RT-DETR
    rtdetr_res = rtdetr_model(
        str(img_path), conf=RTDETR_CONF, iou=IOU_THRESH,
        imgsz=IMG_SIZE, verbose=False,
        augment=True,
    )[0]

    img_w = yolo_res.orig_shape[1]
    img_h = yolo_res.orig_shape[0]

    yolo_boxes,   yolo_scores,   yolo_labels   = extract_boxes_ultralytics(yolo_res,   img_w, img_h)
    rtdetr_boxes, rtdetr_scores, rtdetr_labels = extract_boxes_ultralytics(rtdetr_res, img_w, img_h)

    # RF-DETR
    rfdetr_boxes, rfdetr_scores, rfdetr_labels = [], [], []
    if rfdetr_model is not None:
        try:
            img_pil    = Image.open(img_path).convert("RGB")
            detections = rfdetr_model.predict(img_pil, threshold=RFDETR_CONF)
            rfdetr_boxes, rfdetr_scores, rfdetr_labels = extract_boxes_rfdetr(
                detections, img_w, img_h
            )
        except Exception:
            pass

    # WBF
    all_boxes  = [b for b in [yolo_boxes, rtdetr_boxes, rfdetr_boxes] if b]
    all_scores = [s for s, b in zip([yolo_scores, rtdetr_scores, rfdetr_scores],
                                     [yolo_boxes, rtdetr_boxes, rfdetr_boxes]) if b]
    all_labels = [l for l, b in zip([yolo_labels, rtdetr_labels, rfdetr_labels],
                                     [yolo_boxes, rtdetr_boxes, rfdetr_boxes]) if b]
    all_weights= [w for w, b in zip([YOLO_WEIGHT, RTDETR_WEIGHT, RFDETR_WEIGHT],
                                     [yolo_boxes, rtdetr_boxes, rfdetr_boxes]) if b]

    if not all_boxes:
        continue

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=all_weights,
        iou_thr=WBF_IOU_THR,
        skip_box_thr=WBF_SKIP_BOX,
    )

    for box, score, cls_idx in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h
        category_id = idx_to_real_cat.get(int(cls_idx), -1)
        if category_id == -1:
            continue

        rows.append({
            "annotation_id": annotation_id,
            "image_id"     : image_id,
            "category_id"  : category_id,
            "bbox_x"       : round(float(x1), 2),
            "bbox_y"       : round(float(y1), 2),
            "bbox_w"       : round(float(x2-x1), 2),
            "bbox_h"       : round(float(y2-y1), 2),
            "score"        : round(float(score), 4),
        })
        annotation_id += 1

    print(f"  [{image_id:>6}] {img_path.name} → 누적 {annotation_id-1}개")

# ── CSV 저장 ───────────────────────────────────────────────────
fieldnames = ["annotation_id", "image_id", "category_id",
              "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 60)
print("앙상블 CSV 생성 완료")
print("=" * 60)
print(f"  총 예측 bbox : {len(rows)}개")
print(f"  저장 위치    : {OUTPUT_CSV}")
print("=" * 60)
