# ML Workspace

모델 개발용 작업 영역입니다.

## 개발 환경 및 실행 방법
### 로컬 개발 환경

로컬에서 작업할 때는 가상환경을 만든 뒤 의존성을 설치하는 것을 기본으로 합니다.

#### 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 의존성 설치

```bash
python -m pip install --upgrade pip
python -m pip install -r ml/requirements.txt
```

`ml/requirements.txt`는 Colab 기본 런타임이 제공하는 버전과 CUDA 스택을 최대한 그대로 쓰도록, 상위 패키지만 느슨하게 선언합니다. 로컬 GPU 환경에서는 PyTorch/CUDA 조합을 환경에 맞게 별도로 맞추는 편이 안전합니다.

#### 환경변수 파일 준비

```bash
cp ml/.env.example ml/.env
```

#### Kaggle 원본 데이터 다운로드

`ml/.env`에 들어 있는 `KAGGLE_API_TOKEN`을 읽어 `ml/data/raw`를 채웁니다.

```bash
python -m ml.scripts.00_download_kaggle_data
```

#### 통합 COCO 생성

```bash
python -m ml.scripts.01_merge_coco
```

#### Train/Val COCO split 생성

회전만 다른 동일 샘플군이 train/val에 동시에 들어가지 않도록 파일명 prefix 그룹 단위로 split합니다.

```bash
python -m ml.scripts.02_split_coco
```

#### YOLO 학습용 데이터셋 생성

```bash
python -m ml.scripts.03_build_yolo_dataset
```

#### YOLO 학습 실행

```bash
python -m ml.scripts.04_train_yolo
```

### Colab 실행 환경

Colab에서는 별도 `venv`를 만들지 않고 런타임에 바로 패키지를 설치합니다.

1. Colab 런타임에서 저장소 clone

```bash
!git clone https://github.com/qja0707/OralMedicineImageDetection.git
%cd /content/OralMedicineImageDetection
```

2. 의존성 설치

```bash
!python -m pip install --upgrade pip
!python -m pip install -r ml/requirements.txt
```

3. 환경변수 파일 준비

```bash
!cp ml/.env.example ml/.env
```

복사된 `ml/.env` 파일에 자신의 Kaggle API token을 추가합니다.

4. Kaggle 원본 데이터 다운로드

```bash
!python -m ml.scripts.00_download_kaggle_data
```

5. 통합 COCO 생성

```bash
!python -m ml.scripts.01_merge_coco
```

6. Train/Val COCO split 생성

```bash
!python -m ml.scripts.02_split_coco
```

7. YOLO 학습용 데이터셋 생성

```bash
!python -m ml.scripts.03_build_yolo_dataset
```

8. YOLO 학습 실행

```bash
!python -m ml.scripts.04_train_yolo
```

## 주요 산출물

- `data/interim/merged/train_coco.json`
- `data/interim/merged/category_mapping.json`
- `data/interim/merged/annotation_metadata.json`
- `data/interim/merged/dataset_summary.txt`
- `data/interim/splits/train_coco.json`
- `data/interim/splits/val_coco.json`
- `data/interim/splits/train_val_split.json`
- `data/processed/yolo/images/train`
- `data/processed/yolo/images/val`
- `data/processed/yolo/labels/train`
- `data/processed/yolo/labels/val`
- `data/processed/yolo/pill.yaml`
- `outputs/checkpoints`

## 선택 실행

### annotation 단위 알약 crop dataset이 필요할 때:

```bash
python -m ml.scripts.build_pill_crops
```

- `data/interim/pill_crops/images`
- `data/interim/pill_crops/crop_metadata.json`
- `data/interim/pill_crops/crop_metadata.csv`
- `data/interim/pill_crops/crop_summary.json`

#### 생성 모델 검토용 metadata 정규화가 필요할 때:

```bash
python -m ml.scripts.prepare_pill_generation_dataset
```

- `data/interim/pill_generation/metadata.json`
- `data/interim/pill_generation/metadata.csv`
- `data/interim/pill_generation/summary.json`

#### SDXL + LoRA 학습용 dataset 포맷이 필요할 때:

`pill crop` 이미지와 generation metadata를 바탕으로 `images/`, `captions/`, `metadata.jsonl` 형식의 SDXL 학습 입력을 생성합니다.

```bash
python -m ml.scripts.prepare_sdxl_lora_dataset
```

- `data/interim/sdxl_lora/images`
- `data/interim/sdxl_lora/captions`
- `data/interim/sdxl_lora/metadata.jsonl`
- `data/interim/sdxl_lora/manifest.csv`
- `data/interim/sdxl_lora/summary.json`

#### SDXL + LoRA 학습을 실행할 때:

`diffusers`의 공식 SDXL LoRA 학습 예제 스크립트를 기준으로 `accelerate launch`를 호출합니다.


```bash
python -m ml.scripts.05_train_sdxl_lora
```

기본 설정 파일:

- `ml/configs/train/sdxl_lora.yaml`

기본 출력 경로:

- `ml/outputs/checkpoints/sdxl_lora`

### 학습 완료 후 `best.pt` 기준 검증이 필요할 때만 실행합니다.

```bash
python -m ml.scripts.validate_yolo
```

- `outputs/logs/yolo_validation/low_per_class_metrics.png`

## 하위 디렉터리

- `data/`: 원본, 통합 COCO, split, 최종 학습 데이터
- `notebooks/`: EDA 및 실험 기록
- `src/`: 재사용 가능한 파이프라인 코드
- `configs/`: 경로, 증강, 학습 설정
- `scripts/`: 순차 실행용 엔트리포인트
- `outputs/`: 로그, 그림, 체크포인트

## 기본 흐름

1. `data/raw`에서 원본 확보
2. `src/data/merge_coco.py`로 통합 COCO 생성
3. split/검증 후 `data/interim`에 저장
4. 증강 및 포맷 변환으로 `data/processed` 생성
5. 학습 결과를 `outputs`와 `../models`에 저장
