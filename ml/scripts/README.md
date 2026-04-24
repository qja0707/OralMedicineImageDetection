# Scripts

여기에는 순서대로 실행하는 래퍼 스크립트를 둡니다.

공통 전제:

- `ml/requirements.txt` 설치 후 실행
- Kaggle이 필요한 스크립트는 `ml/.env`를 직접 읽음

현재 사용 중:

- `00_download_kaggle_data.py`: `ml/.env`에서 `KAGGLE_API_TOKEN`을 읽어 Kaggle competition 데이터를 다운로드하고 `ml/data/raw`를 채움
- `01_merge_coco.py`: `ml/data/raw`의 원본 annotation/image를 읽어 `ml/data/interim/merged`에 통합 COCO를 생성
- `02_split_coco.py`: 회전만 다른 동일 샘플군이 train/val에 동시에 들어가지 않도록 그룹 단위로 COCO split을 생성
- `03_build_yolo_dataset.py`: split된 COCO를 기준으로 `ml/data/processed/yolo` 아래 YOLO txt 데이터셋과 `pill.yaml`을 생성
- `04_train_yolo.py`: YOLO 데이터셋과 증강/학습 설정을 읽어 Ultralytics 학습을 실행

선택 실행:

- `validate_yolo.py`: 학습 완료 후 `best.pt`와 `pill.yaml`을 기준으로 `model.val()`을 수행하고 낮은 클래스부터 matplotlib 그래프로 저장

예시:

- `01_merge_coco.py`
- `02_validate_coco.py`
- `03_split_coco.py`
- `04_preview_augmentation.py`
- `05_build_yolo_dataset.py`
- `06_train_yolo.py`
