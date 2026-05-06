"""PillScope 백엔드 — FastAPI"""
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
import numpy as np
import os
from pathlib import Path
from PIL import Image
import io
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "models" / "best.pt"
CLASS_MAPPING_PATH = BASE_DIR / "models" / "real_category_mapping.json"
PILL_DB_PATH = BASE_DIR / "pill_info.json"
CONFIDENCE = 0.40

# 약품 DB
pill_db = {}
index_to_catid = {}
class_mapping = {}
if PILL_DB_PATH.exists():
    with open(PILL_DB_PATH, encoding="utf-8") as f:
        pill_db = json.load(f)
    sorted_ids = sorted(pill_db.keys(), key=lambda x: int(x))
    index_to_catid = {i: cid for i, cid in enumerate(sorted_ids)}

if CLASS_MAPPING_PATH.exists():
    with open(CLASS_MAPPING_PATH, encoding="utf-8") as f:
        mapping_dict = json.load(f)["idx_to_real_category_id"]
        class_mapping = {int(k): str(v) for k, v in mapping_dict.items()}

# 모델
model = None
def load_model():
    global model
    if MODEL_PATH.exists():
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        dummy = np.zeros((640, 480, 3), dtype=np.uint8)
        model(dummy, imgsz=640, conf=0.99, verbose=False)
load_model()


def resolve_pill_info(cls_id):
    cid = class_mapping.get(cls_id) or index_to_catid.get(cls_id)
    if cid and cid in pill_db:
        return pill_db[cid]
    if model:
        name = model.names.get(cls_id, "")
        for c, info in pill_db.items():
            if info["name"] == name:
                return info
    return None


app = FastAPI()

# 정적 파일 (HTML/CSS/JS)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
async def root():
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    results = model(img_array, imgsz=1280, conf=CONFIDENCE, verbose=False)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        info = resolve_pill_info(cls_id)
        name = info["name"] if info else model.names.get(cls_id, f"Class {cls_id}")

        det = {
            "name": name,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2],
        }
        if info:
            det["info"] = info

        detections.append(det)

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # 검출 이미지 (약 이름만 표시)
    from PIL import ImageDraw, ImageFont
    annotated_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(annotated_pil)
    try:
        font = ImageFont.truetype("malgunbd.ttf", 40)
    except OSError:
        try:
            font = ImageFont.truetype("malgun.ttf", 40)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except OSError:
                font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        name = det["name"]
        draw.rectangle([x1, y1, x2, y2], outline="#4ECDC4", width=5)
        tw = draw.textlength(name, font=font) if hasattr(draw, 'textlength') else len(name) * 22
        draw.rectangle([x1, y1 - 52, x1 + tw + 16, y1], fill="white", outline="#4ECDC4", width=2)
        draw.text((x1 + 8, y1 - 50), name, fill="#2D3436", font=font)

    import base64
    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({
        "detections": detections,
        "image": img_b64,
        "count": len(detections),
    })


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "pills": len(pill_db),
        "class_mapping": CLASS_MAPPING_PATH.name if class_mapping else "pill_info_numeric_order",
        "mapped_classes": len(class_mapping) if class_mapping else len(index_to_catid),
    }


# Gemini AI 약사 채팅
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

PHARMACIST_PROMPTS = {
    "kim": "당신은 베테랑 약사 김원장입니다. 30년 경력의 꼼꼼하고 신중한 약사입니다. 항상 정확한 정보를 제공하며, 존댓말을 사용합니다. 환자의 건강을 최우선으로 생각합니다.",
    "lee": "당신은 젊은 약사 이준입니다. 친절하고 편안한 말투로 상담합니다. 어려운 의학 용어를 쉽게 풀어서 설명합니다. 이모티콘을 가끔 사용합니다.",
    "park": "당신은 따뜻한 약사 박미소입니다. 어르신들에게 '어머님', '아버님'으로 호칭하며, 따뜻하고 다정한 말투를 사용합니다. 쉬운 말로 천천히 설명합니다.",
    "choi": "당신은 전문가 약사 최유진입니다. 정확하고 체계적으로 답변합니다. 근거 기반으로 설명하며, 필요시 주의사항을 꼭 언급합니다.",
}

class ChatRequest(BaseModel):
    message: str
    pharmacist: str = "kim"
    history: list = []
    user_symptoms: list = []
    current_meds: list = []

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not GEMINI_KEY:
        return JSONResponse({"reply": "AI 약사 기능을 사용하려면 GEMINI_API_KEY 환경변수를 설정해주세요."}, status_code=200)
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_KEY)

        system_prompt = PHARMACIST_PROMPTS.get(req.pharmacist, PHARMACIST_PROMPTS["kim"])
        system_prompt += "\n\n[역할] 당신은 약에 대한 상담을 하는 AI 약사입니다."
        system_prompt += "\n[필수 규칙]"
        system_prompt += "\n1. '병원에 가세요', '의사에게 상담하세요' 같은 회피성 답변은 절대 하지 않습니다."
        system_prompt += "\n2. 환자가 증상을 말하면 먼저 증상을 세분화하는 질문을 합니다. (예: '아랫배인가요, 윗배인가요?', '식후에 아프신가요?')"
        system_prompt += "\n3. 충분한 정보가 모이면 구체적인 약 이름을 추천합니다."
        system_prompt += "\n4. 추천 시 효능, 복용법, 주의사항을 함께 안내합니다."
        system_prompt += "\n5. 환자의 기저질환과 현재 복용 약을 반드시 고려합니다."
        system_prompt += "\n6. 대화를 이어가며 환자의 상태를 파악합니다. 한 번에 끝내지 않습니다."
        system_prompt += "\n7. 답변 마지막에 항상 후속 질문을 합니다."

        if req.user_symptoms:
            system_prompt += f"\n\n환자의 기저질환: {', '.join(req.user_symptoms)}. 이를 반드시 고려하여 약 추천 및 주의사항을 안내하세요."

        if req.current_meds:
            system_prompt += f"\n현재 복용 중인 약: {', '.join(req.current_meds)}. 새로 추천하는 약과의 병용금기를 반드시 확인하고 안내하세요."

        full_prompt = system_prompt + "\n\n"
        for h in req.history[-10:]:
            role = "환자" if h["type"] == "user" else "약사"
            full_prompt += f"{role}: {h['text']}\n"
        full_prompt += f"환자: {req.message}\n약사:"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config={"temperature": 0.7, "max_output_tokens": 4096, "thinking_config": {"thinking_budget": 0}},
        )

        return JSONResponse({"reply": response.text.strip()})

    except Exception as e:
        return JSONResponse({"reply": f"죄송합니다. 현재 상담이 어렵습니다. ({str(e)[:50]})"}, status_code=200)
