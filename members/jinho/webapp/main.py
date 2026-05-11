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

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
PILL_DB_PATH = BASE_DIR / "pill_info.json"
CONFIDENCE = 0.40

# 약품 DB
pill_db = {}
index_to_catid = {}
if PILL_DB_PATH.exists():
    with open(PILL_DB_PATH, encoding="utf-8") as f:
        pill_db = json.load(f)
    sorted_ids = sorted(pill_db.keys(), key=lambda x: int(x))
    index_to_catid = {i: cid for i, cid in enumerate(sorted_ids)}

# 모델
model = None
current_model_name = ""

def load_model(name="best.pt"):
    global model, current_model_name
    path = MODELS_DIR / name
    if path.exists():
        from ultralytics import YOLO
        model = YOLO(str(path))
        dummy = np.zeros((640, 480, 3), dtype=np.uint8)
        model(dummy, imgsz=640, conf=0.99, verbose=False)
        current_model_name = name

def get_model_list():
    if not MODELS_DIR.exists():
        return []
    return sorted([f.name for f in MODELS_DIR.iterdir() if f.suffix == ".pt"])

load_model()


def resolve_pill_info(cls_id):
    cid = index_to_catid.get(cls_id)
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
    FONT_SIZE = 20
    try:
        font = ImageFont.truetype("malgunbd.ttf", FONT_SIZE)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", FONT_SIZE)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
            except OSError:
                font = ImageFont.load_default()

    label_h = FONT_SIZE + 10
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        name = det["name"]
        draw.rectangle([x1, y1, x2, y2], outline="#4ECDC4", width=3)
        tw = draw.textlength(name, font=font) if hasattr(draw, 'textlength') else len(name) * 12
        draw.rectangle([x1, y1 - label_h, x1 + tw + 12, y1], fill="white", outline="#4ECDC4", width=2)
        draw.text((x1 + 6, y1 - label_h + 3), name, fill="#2D3436", font=font)

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
    return {"status": "ok", "model_loaded": model is not None, "pills": len(pill_db)}


@app.get("/api/models")
async def list_models():
    models = get_model_list()
    return JSONResponse({"models": models, "current": current_model_name})


class ModelSwitch(BaseModel):
    name: str

@app.post("/api/models/switch")
async def switch_model(req: ModelSwitch):
    path = MODELS_DIR / req.name
    if not path.exists() or path.suffix != ".pt":
        return JSONResponse({"error": "모델을 찾을 수 없습니다"}, status_code=404)
    try:
        load_model(req.name)
        return JSONResponse({"success": True, "current": current_model_name})
    except Exception as e:
        return JSONResponse({"error": f"모델 로드 실패: {str(e)[:80]}"}, status_code=500)


# 처방전 OCR
ocr_reader = None
def load_ocr():
    global ocr_reader
    try:
        import easyocr
        ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
    except Exception:
        pass

@app.post("/api/ocr")
async def ocr_prescription(file: UploadFile = File(...)):
    if ocr_reader is None:
        load_ocr()
    if ocr_reader is None:
        return JSONResponse({"error": "OCR 모듈을 로드할 수 없습니다"}, status_code=500)

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    results = ocr_reader.readtext(img_array)

    ocr_texts = [text for _, text, conf in results if conf > 0.3]
    all_text = ' '.join(ocr_texts)

    matched = []
    for cid, info in pill_db.items():
        name = info["name"]
        short = name.split("(")[0].replace(" ", "")
        if len(short) < 2:
            continue
        for ocr_t in ocr_texts:
            clean = ocr_t.replace(" ", "")
            if short[:4] in clean or clean in short:
                if cid not in [m["catId"] for m in matched]:
                    matched.append({"catId": cid, "name": name, "info": info, "ocrText": ocr_t})
                break

    return JSONResponse({
        "ocrTexts": ocr_texts,
        "matched": matched,
        "count": len(matched),
    })


# Gemini AI 약사 채팅
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

PHARMACIST_PROMPTS = {
    "kim": "당신은 베테랑 약사 김원장입니다. 30년 경력의 꼼꼼하고 신중한 약사입니다. 항상 정확한 정보를 제공하며, 존댓말을 사용합니다. 환자의 건강을 최우선으로 생각합니다.",
    "lee": "당신은 젊은 약사 이준입니다. 친절하고 편안한 말투로 상담합니다. 어려운 의학 용어를 쉽게 풀어서 설명합니다. 이모티콘을 가끔 사용합니다.",
    "park": "당신은 따뜻한 약사 박미소입니다. 따뜻하고 다정한 말투를 사용합니다. 쉬운 말로 천천히 설명합니다. 환자의 이름이 주어지면 '님'을 붙여 호칭하고, 이름이 없으면 '환자분'으로 호칭합니다.",
    "choi": "당신은 전문가 약사 최유진입니다. 정확하고 체계적으로 답변합니다. 근거 기반으로 설명하며, 필요시 주의사항을 꼭 언급합니다.",
}

class ChatRequest(BaseModel):
    message: str
    pharmacist: str = "kim"
    history: list = []
    user_symptoms: list = []
    current_meds: list = []
    user_name: str = ""

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
        system_prompt += "\n8. 약 이름을 언급할 때는 반드시 **약이름** 형태로 볼드 표시합니다. (예: **마그밀정**, **타이레놀정**)"
        system_prompt += "\n9. 상담 중 증상에 맞는 건강기능식품도 자연스럽게 추천합니다. 고혈압→코큐텐/홍국, 당뇨→바나바잎/여주, 간→밀크시슬, 위장→프로바이오틱스/매스틱, 알레르기→퀘르세틴, 임산부→프리나탈. 예: '참고로 **코큐텐**도 혈압 관리에 도움이 됩니다. 홈 화면 건강식품 탭에서 확인해보세요!' 처럼 대화 끝에 슬쩍 언급합니다."
        system_prompt += "\n10. 건강기능식품 추천 시 너무 강하게 권하지 말고 '참고로', '혹시 관심 있으시면', '도움이 될 수 있는' 같은 부드러운 표현을 사용합니다."

        if req.user_name:
            system_prompt += f"\n\n환자 이름: {req.user_name}. 이름을 불러주세요."

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

        reply_text = response.text.strip()

        SUPPLEMENT_MAP = {
            "코큐텐": {"name": "코큐텐", "tag": "심장·혈압 건강", "img": "/static/img/supplements/코큐텐.png"},
            "홍국": {"name": "홍국", "tag": "콜레스테롤 관리", "img": "/static/img/supplements/홍국.png"},
            "은행잎": {"name": "은행잎 추출물", "tag": "혈행 개선", "img": "/static/img/supplements/은행잎.png"},
            "바나바": {"name": "바나바잎", "tag": "혈당 관리", "img": "/static/img/supplements/바나바.png"},
            "여주": {"name": "여주 추출물", "tag": "천연 인슐린", "img": "/static/img/supplements/여주.png"},
            "상엽": {"name": "상엽 추출물", "tag": "식후 혈당 조절", "img": "/static/img/supplements/상엽.png"},
            "밀크시슬": {"name": "밀크시슬", "tag": "간세포 보호", "img": "/static/img/supplements/밀크시슬.png"},
            "밀크씨슬": {"name": "밀크시슬", "tag": "간세포 보호", "img": "/static/img/supplements/밀크시슬.png"},
            "씨슬파워": {"name": "씨슬파워", "tag": "간 해독 강화", "img": "/static/img/supplements/씨슬파워.png"},
            "민들레": {"name": "민들레 추출물", "tag": "간 기능 개선", "img": "/static/img/supplements/민들레엑스.png"},
            "키드니포뮬러": {"name": "키드니포뮬러", "tag": "신장 기능 보호", "img": "/static/img/supplements/키드니포믈러.png"},
            "네틀포스": {"name": "네틀포스", "tag": "이뇨·신장 건강", "img": "/static/img/supplements/네틀포스.png"},
            "코디세핀": {"name": "코디세핀", "tag": "신장 세포 보호", "img": "/static/img/supplements/코디세핀.png"},
            "프로바이오틱스": {"name": "프로바이오틱스", "tag": "장내 유익균", "img": "/static/img/supplements/프로바이오틱스.png"},
            "매스틱": {"name": "매스틱", "tag": "위점막 보호", "img": "/static/img/supplements/매스틱.png"},
            "퀘르세틴": {"name": "퀘르세틴", "tag": "항히스타민 효과", "img": "/static/img/supplements/퀘르세틴.png"},
            "프리나탈": {"name": "프리나탈", "tag": "엽산·철분 복합", "img": "/static/img/supplements/프리나탈.png"},
            "멀티비타민": {"name": "멀티비타민", "tag": "종합 영양 보충", "img": "/static/img/supplements/멀티비타민.png"},
            "헛깨": {"name": "헛깨 추출물", "tag": "간 건강·숙취", "img": "/static/img/supplements/헛깨.png"},
        }

        matched_supplements = []
        seen = set()
        for keyword, info in SUPPLEMENT_MAP.items():
            if keyword in reply_text and info["name"] not in seen:
                matched_supplements.append(info)
                seen.add(info["name"])

        return JSONResponse({"reply": reply_text, "supplements": matched_supplements})

    except Exception as e:
        return JSONResponse({"reply": f"죄송합니다. 현재 상담이 어렵습니다. ({str(e)[:50]})", "supplements": []}, status_code=200)
