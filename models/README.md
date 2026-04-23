# Models

이 디렉터리는 학습된 모델 파일을 로컬에 배치하는 용도입니다.

## expected files

- `models/yolo/best.pt`
- `models/yolo/last.pt`
- `models/exported/model.onnx`

## usage

- 학습 결과물은 직접 다운로드해서 위 경로에 배치합니다.
- 대용량 모델 파일은 Git에 커밋하지 않습니다.
- 앱과 추론 코드는 위 기본 경로를 기준으로 모델을 찾도록 맞춥니다.

## notes

- 파일명이 바뀌면 앱 또는 추론 스크립트의 로드 경로도 함께 수정해야 합니다.
- 실험별 모델은 별도 하위 폴더를 두거나 버전명을 파일명에 포함해서 관리합니다.
