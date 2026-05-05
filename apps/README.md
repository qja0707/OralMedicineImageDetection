# Apps

이 디렉터리는 학습된 모델을 실제로 사용하는 서비스 코드를 둡니다.

- `main.py`: 현재 서비스 진입점
- `static/`: 프론트엔드 정적 파일과 샘플 자산
- `requirements.txt`: 앱 전용 의존성
- `Dockerfile`: 앱 컨테이너 빌드 설정
- 모델 weight는 직접 커밋하지 말고 `models/`를 참조하도록 유지
- 앱 전용 의존성은 앱 내부 `requirements.txt` 또는 추후 별도 환경 파일로 관리

## 실행 방법

현재 앱은 FastAPI 기반의 PillScope 웹 서비스입니다. 정적 프론트엔드는 `/`에서 제공되고, 약 이미지 탐지는 `/api/detect`, 상태 확인은 `/api/health`에서 제공합니다.

### 1. 모델 파일 준비

약 이미지 탐지 기능을 사용하려면 학습된 YOLO weight 파일을 아래 경로에 둡니다.

```bash
apps/models/best.pt
```

모델 파일이 없으면 서버는 실행되지만 `/api/detect`는 `Model not loaded` 오류를 반환합니다.

### 2. 로컬 실행

저장소 루트에서 다음 명령을 실행합니다.

```bash
cd apps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:7860
```

상태 확인:

```bash
curl http://localhost:7860/api/health
```

### 3. AI 약사 채팅 사용

AI 약사 채팅 기능은 `apps/.env` 또는 시스템 환경변수에 `GEMINI_API_KEY`가 있을 때 동작합니다.

`apps/.env` 파일을 만들고 아래처럼 입력합니다.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

그 다음 평소와 같이 서버를 실행합니다.

```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

환경변수를 설정하지 않아도 앱은 실행되며, 채팅 API는 키 설정 안내 메시지를 반환합니다.

### 4. Docker 실행

`apps` 디렉터리에서 이미지를 빌드하고 실행합니다.

```bash
cd apps
docker build -t pillscope-app .
docker run --rm -p 7860:7860 pillscope-app
```

Gemini API 키를 함께 넘기려면 다음처럼 실행합니다.

```bash
docker run --rm -p 7860:7860 --env-file .env pillscope-app
```

Docker 이미지 빌드 시점에 `apps/models/best.pt`가 있으면 모델 파일도 이미지에 포함됩니다. 모델을 이미지에 포함하지 않고 실행 시점에 마운트하려면 저장소 루트에서 다음처럼 실행합니다.

```bash
docker run --rm -p 7860:7860 \
  -v "$(pwd)/apps/models:/app/models:ro" \
  --env-file apps/.env \
  pillscope-app
```
