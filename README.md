# OralMedicineImageDetection

## 프로젝트 개요
AI 모델을 활용하여 경구약제 객체를 탐지하는 프로젝트입니다. 이미지 데이터에서 최대 4개의 경구약제를 식별하고 위치를 검출하는 객체 탐지 모델을 구현과 성능 개선을 목표로 합니다.

## 프로젝트 목표
경구약제 이미지에서 객체를 정확하게 탐지할 수 있는 AI 모델을 구축하고, 이를 통해 약제 식별 자동화 가능성을 확인합니다.

---

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

#### 환경변수 파일 준비

```bash
cp ml/.env.example ml/.env
```

#### Kaggle 원본 데이터 다운로드

`ml/.env`에 들어 있는 `KAGGLE_API_TOKEN`을 읽어 `ml/data/raw`를 채웁니다.

```bash
python -m ml.scripts.00_download_kaggle_data
```

#### 통합 COCO 생성 스크립트 실행

```bash
python -m ml.scripts.01_merge_coco
```

#### Train/Val COCO split 생성

회전만 다른 동일 샘플군이 train/val에 동시에 들어가지 않도록 파일명 prefix 그룹 단위로 split합니다.

```bash
python -m ml.scripts.02_split_coco
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

복사된 ml/.env 파일에 자신의 kaggle api token 추가

4. Kaggle 원본 데이터 다운로드

```bash
!python -m ml.scripts.00_download_kaggle_data
```

5. 통합 COCO 생성 스크립트 실행

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

### 산출물 확인

- `ml/data/interim/merged/train_coco.json`
- `ml/data/interim/merged/category_mapping.json`
- `ml/data/interim/merged/annotation_metadata.json`
- `ml/data/interim/merged/dataset_summary.txt`
- `ml/data/interim/splits/train_coco.json`
- `ml/data/interim/splits/val_coco.json`
- `ml/data/interim/splits/train_val_split.json`
- `ml/data/processed/yolo/images/train`
- `ml/data/processed/yolo/images/val`
- `ml/data/processed/yolo/labels/train`
- `ml/data/processed/yolo/labels/val`
- `ml/data/processed/yolo/pill.yaml`


---

## 일정
2026년 4월 20일 ~ 2026년 5월 11일

## 팀원 및 역할
| 역할 | 이름 |
| --- | --- |
| Project Manager | 안은남 |
| Experimentation Lead | 박재철 |
| Interface Lead | 이진호 |
| Data Engineer/Model Architect | 하태진, 박규범 |

## 결과
추후 작성 예정

## 디렉터리 구조

```text
apps/
  main.py              모델 추론 결과를 사용하는 서비스 진입점
  static/              서비스 정적 자산
ml/
  data/                원본, 중간 산출물, 학습용 데이터셋
  notebooks/           EDA 및 실험 노트북
  src/                 데이터 전처리, 증강, 학습 코드
  configs/             데이터/학습 설정
  scripts/             순차 실행용 진입 스크립트
  outputs/             로그, 시각화, 체크포인트
models/
  yolo/                학습된 weight
  exported/            ONNX/TorchScript 등 배포용 모델
shared/
  schemas/             앱과 ML이 공유하는 스키마
  utils/               공용 유틸리티
```

## 작업 원칙

- 원본 데이터는 `ml/data/raw`에만 둡니다.
- 통합 COCO와 split 산출물은 `ml/data/interim`에서 관리합니다.
- 최종 학습 포맷은 `ml/data/processed`에 생성합니다.
- 서비스 코드는 `apps/inference_service`에서만 관리합니다.
- 기존 개인 작업물은 `members/`에 유지하고, 공용 작업은 새 루트 구조를 기준으로 이어갑니다.
