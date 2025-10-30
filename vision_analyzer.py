import base64
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI
from PIL import Image

# --------------------------------------------------
# ✅ 환경변수 로드 (.env를 프로젝트 루트에서 자동 탐색)
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent  # cleanbin 폴더 기준
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY2")
if not OPENAI_API_KEY:
    raise RuntimeError(f"환경 변수 OPENAI_API_KEY2가 설정되지 않았습니다. (.env: {ENV_PATH})")

client = OpenAI(api_key=OPENAI_API_KEY)

# 모델과 설정
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_SIDE = 1280
PROMPT = (
    "다음 지침을 엄격히 따르세요. 이 이미지는 재활용 쓰레기 또는 생활폐기물 사진입니다. "
    "이미지에 보이는 사물과 재질을 한국어 '명사' 형태로만 추출하여 정확한 JSON 객체 하나로만 응답하세요. "
    "출력 형식은 정확히 다음과 같아야 합니다: {\"items\": [\"플라스틱병\", \"종이상자\", \"유리병\"]} "
    "각 배열 요소는 명사(띄어쓰기 없음 또는 최소화)로 작성하고 중복은 제거하세요. "
    "설명, 문장, 불필요한 기호나 추가 텍스트는 절대 포함하지 마세요. "
    "불확실한 항목은 그대로 포함하되 추정 표시(예: '~추정')는 붙이지 마세요."
)

# FastAPI Router 생성
router = APIRouter(prefix="/vision", tags=["GPT Vision Analyzer"])


def image_bytes_to_data_url(b: bytes, ctype: str = "image/png") -> str:
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{ctype};base64,{b64}"


def call_gpt_vision(data_url: str, model: str = DEFAULT_MODEL) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    return response.output_text.strip()


@router.post("/analyze")
def analyze_image(file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)):
    """
    업로드된 PNG/JPG 이미지를 분석해 GPT-4 Vision의 설명을 반환합니다.
    """
    try:
        content = file.file.read()
        img = Image.open(BytesIO(content)).convert("RGB")

        # 크기 제한
        w, h = img.size
        scale = max(w, h) / float(MAX_SIDE)
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

        # PNG/JPG 형식 유지
        fmt = "PNG" if file.filename.lower().endswith(".png") else "JPEG"
        ctype = "image/png" if fmt == "PNG" else "image/jpeg"

        bio = BytesIO()
        img.save(bio, format=fmt, quality=90 if fmt == "JPEG" else None)

        data_url = image_bytes_to_data_url(bio.getvalue(), ctype=ctype)
        result_text = call_gpt_vision(data_url, model=model)

        return JSONResponse({"result": result_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
