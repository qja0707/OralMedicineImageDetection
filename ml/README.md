# ML Workspace

모델 개발용 작업 영역입니다.

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
