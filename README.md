# OralMedicineImageDetection

## 프로젝트 개요
AI 모델을 활용하여 경구약제 객체를 탐지하는 프로젝트입니다. 이미지 데이터에서 최대 4개의 경구약제를 식별하고 위치를 검출하는 객체 탐지 모델을 구현과 성능 개선을 목표로 합니다.

## 프로젝트 목표
경구약제 이미지에서 객체를 정확하게 탐지할 수 있는 AI 모델을 구축하고, 이를 통해 약제 식별 자동화 가능성을 확인합니다.

---

## 개발 환경 및 실행 방법
### Apps

서비스 실행 및 앱 개발 관련 내용은 [apps/README.md](apps/README.md)를 참고합니다.

### ML

데이터 다운로드, COCO 병합, split, YOLO 데이터셋 생성, 학습/검증 등 ML 파이프라인은 [ml/README.md](ml/README.md)를 참고합니다.


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
