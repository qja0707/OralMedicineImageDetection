"""PillScope 백엔드 — FastAPI"""
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
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
def load_model():
    global model
    if MODEL_PATH.exists():
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        dummy = np.zeros((640, 480, 3), dtype=np.uint8)
        model(dummy, imgsz=640, conf=0.99, verbose=False)
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
    return {"status": "ok", "model_loaded": model is not None, "pills": len(pill_db)}
