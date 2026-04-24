# Apps

이 디렉터리는 학습된 모델을 실제로 사용하는 서비스 코드를 둡니다.

- `main.py`: 현재 서비스 진입점
- `static/`: 프론트엔드 정적 파일과 샘플 자산
- `requirements.txt`: 앱 전용 의존성
- `Dockerfile`: 앱 컨테이너 빌드 설정
- 모델 weight는 직접 커밋하지 말고 `models/`를 참조하도록 유지
- 앱 전용 의존성은 앱 내부 `requirements.txt` 또는 추후 별도 환경 파일로 관리
