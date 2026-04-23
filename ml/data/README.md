# Data Layout

- `raw/`: 원본 이미지와 원본 annotation
- `interim/merged/`: 통합 COCO와 category mapping
- `interim/splits/`: train/val/test split 결과
- `interim/metadata/`: 데이터 통계, 메타데이터
- `processed/coco/`: 학습용 COCO split
- `processed/yolo/`: YOLO 포맷 데이터셋
- `processed/samples/augmentation_preview/`: 증강 확인용 샘플 이미지

원본 JSON은 최대한 수정하지 않고, 파생 산출물을 분리해서 관리합니다.
