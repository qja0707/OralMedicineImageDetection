# 헬스잇 — 경구약제 이미지 객체 검출

> **[AI10] 경구약제 이미지 객체 검출 대회** | Kaggle mAP@[0.75:0.95] = **0.93552**

알약 사진 한 장으로 약품명·성분·효능·주의사항을 즉시 확인하는 AI 탐지 시스템입니다.

---

## 프로젝트 구조

```
📦 초급 프로젝트/
├── output/
│   ├── pill.yaml                   # YOLO 학습 설정
│   ├── train_coco.json             # COCO 형식 통합 라벨
│   ├── category_mapping.json       # 클래스 인덱스 매핑
│   ├── real_category_mapping.json  # 실제 category_id 매핑
│   └── dataset/
│       ├── images/
│       │   ├── train/              # 학습 이미지
│       │   └── val/                # 검증 이미지 (46장)
│       └── labels/
│           ├── train/              # YOLO txt 라벨
│           └── val/
├── internet_augment.py             # 인터넷 이미지 파란 배경 합성
├── minority_augment.py             # 소수 클래스 Crop & Paste 증강
├── pseudo_labeling.py              # Pseudo Labeling 재학습
├── make_submission.py              # 단일 모델 제출 CSV 생성
├── ensemble_submission.py          # 3모델 WBF 앙상블 제출
├── requirements.txt
└── README.md
```

---

## 실험 결과 요약

| 실험 | 모델 | 데이터 | mAP@0.75:0.95 | Kaggle |
|------|------|--------|--------------|--------|
| 1차 | YOLOv11s/m, RT-DETRv2 | 186장 | - | - |
| 2차 | YOLOv11l | 186장 | - | - |
| 3차 | YOLOv8l, YOLOv11l | 1026장 | 0.919 | - |
| 4차 | YOLOv11l | 1866장 | 0.916 | - |
| **5차 ★** | **YOLOv11l** | **1116장** | **0.936** | **0.863** |
| 6차 | RT-DETRv2 | 1206장 | 0.883 | - |
| 7차 | RF-DETRv3 | 186장 | 0.882 | - |
| 앙상블 | YOLO+RTDETR+RFDETR WBF | - | - | **0.93552** |

---

## 환경 설정

```bash
# 가상환경 생성
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# 패키지 설치
pip install -r requirements.txt
```

---

## 데이터 증강

### 1. Minority Augment (소수 클래스 증강)
```bash
python minority_augment.py
# 186장 → 1026장 (소수 클래스 Crop & Paste)
```

### 2. 인터넷 이미지 합성
```bash
python internet_augment.py
# 파란 배경에 알약 RGBA 이미지 합성
# 1026장 → 1116~1206장
```

---

## 모델 학습

### YOLOv11l (메인 모델)
```bash
# 5차 실험 기준 (최고 성능)
from ultralytics import YOLO
model = YOLO("yolo11l.pt")
model.train(
    data="output/pill.yaml",
    epochs=150,
    imgsz=1280,
    batch=4,
    device=0,
)
```

---

## 제출 파일 생성

### 단일 모델 제출
```bash
python make_submission.py
```

### 3모델 WBF 앙상블 제출 (최고 점수)
```bash
python ensemble_submission.py
# YOLOv11l + RT-DETRv2 + RF-DETRv3 WBF
# conf=0.01, iou_thr=0.55, weight=1:1:1
# → Kaggle mAP 0.93552
```

---

## 모델 가중치

대용량 파일로 GitHub에 포함되지 않습니다.

| 모델 | 파일명 | 비고 |
|------|--------|------|
| YOLOv11l 5차 | `exp_yolo11l_final/weights/best.pt` | 단일 최고 성능 |
| RT-DETRv2 | `exp_rtdetr_retrain/weights/best_1206.pt` | 앙상블 사용 |
| RF-DETRv3 | `exp_rfdetr_v3/checkpoint_best_regular.pth` | 앙상블 사용 |

---

## 팀 구성

| 이름 | 담당 |
|------|------|
| 은남 | 서론, EDA, 결론 |
| 규범 | 데이터 전처리, 증강 |
| 재철 | 모델 실험, 앙상블, 성능 분석 |
| 진호 | 어플리케이션 (PillScope) |

---

## 어플리케이션

**PillScope** — HuggingFace Spaces 배포

카메라 한 장으로 알약의 이름·성분·효능·주의사항까지 즉시 확인하는 모바일 웹 플랫폼
