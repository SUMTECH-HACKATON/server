# --------------------------------------------------
# vision_router.py (Hybrid RAG + Upcycling Tips 버전)
# --------------------------------------------------
import base64
import json
import math
import os
import re
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI
from PIL import Image

# ✅ embedding_util 임포트
from embedding_util import get_text_embeddings

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
EMBED_PATH = ROOT_DIR / "data" / "embedding.json"  # ✅ RAG 데이터 경로

PROMPT_VISION = (
    """
    다음 재활용품 사진이 무슨 물건인지 json으로 명사로 반환해줘.
    출력 형식 예시는 다음과 같아야 해:
    {"items": ["플라스틱병", "종이상자", "유리병", "종이컵"],
    "materials": ["비닐류", "플라스틱", "종이", "유리", "금속", "고철","스티로폼"],
    "details": ["뚜껑이 있는 플라스틱병", "박스 형태의 종이상자", "투명한 유리병", "지저분한 종이컵"]}
    반드시 JSON 형식으로만 응답해줘.
    """
)

PROMPT_RAG_GUIDE = (
    """
    당신은 한국의 재활용 전문가입니다.
    아래는 데이터베이스에서 검색된 유사 품목의 재활용 가이드입니다.
    이 내용을 참고하되 그대로 복사하지 말고, 자연스럽고 일관성 있게 정리해 주세요.

    각 품목별로 단계별 절차를 한국어 리스트 형태로 반환하세요.
    예시:
    [
      "#1. 플라스틱병은 내용물을 비우세요.",
      "#2. 라벨과 뚜껑을 제거하세요.",
      "#3. 플라스틱 전용 수거함에 넣으세요."
    ]

    반드시 JSON 형식의 리스트로만 응답하세요.
    """
)

PROMPT_UPCYCLING_TIP = (
    """
    당신은 환경 친화적인 업사이클링 전문가입니다.
    주어진 재활용품 목록을 기반으로 각 품목을 창의적으로 재활용하거나 재사용할 수 있는 아이디어를 만들어 주세요.
    각 품목당 하나씩, 간단한 문장으로 리스트를 반환하세요.
    예시:
    ["플라스틱병은 물뿌리개나 화분으로 재활용할 수 있습니다.",
     "유리병은 조명 장식이나 꽃병으로 사용할 수 있습니다."]

    반드시 JSON 형식의 리스트로만 응답하세요.
    """
)

router = APIRouter(prefix="/vision", tags=["GPT Vision Analyzer"])


# --------------------------------------------------
# 🔹 유틸 함수들
# --------------------------------------------------
def image_bytes_to_data_url(b: bytes, ctype: str = "image/png") -> str:
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{ctype};base64,{b64}"


def cosine_similarity(v1, v2):
    """두 벡터 간 코사인 유사도 계산."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2 + 1e-9)


def load_embedding_data(path: Path):
    """embedding.json 로드."""
    if not path.exists():
        raise FileNotFoundError(f"임베딩 파일이 존재하지 않습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("embeddings", [])


def call_gpt_vision(data_url: str, model: str = DEFAULT_MODEL) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT_VISION},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    return response.output_text.strip()


# --------------------------------------------------
# 🔹 Vision + Hybrid RAG + Upcycling Tips 엔드포인트
# --------------------------------------------------
@router.post("/analyze")
def analyze_image(file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)):
    """이미지를 분석하고 GPT가 RAG 정보를 참고해 재활용 절차와 업사이클링 팁을 생성."""
    try:
        # -----------------------------
        # 1️⃣ 이미지 전처리
        # -----------------------------
        content = file.file.read()
        img = Image.open(BytesIO(content)).convert("RGB")

        w, h = img.size
        scale = max(w, h) / float(MAX_SIDE)
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

        fmt = "PNG" if file.filename.lower().endswith(".png") else "JPEG"
        ctype = "image/png" if fmt == "PNG" else "image/jpeg"
        bio = BytesIO()
        img.save(bio, format=fmt, quality=90 if fmt == "JPEG" else None)
        data_url = image_bytes_to_data_url(bio.getvalue(), ctype=ctype)

        # -----------------------------
        # 2️⃣ Vision으로 품목 추출
        # -----------------------------
        result_text = call_gpt_vision(data_url, model=model)
        match = re.search(r"\{[\s\S]*\}", result_text)
        if not match:
            raise ValueError("No JSON object found in GPT response")

        parsed = json.loads(match.group(0))
        items = parsed.get("items", [])
        if not items:
            return JSONResponse(
                {
                    "result": {
                        "message": "잘못된 이미지입니다.",
                        "items": [],
                        "disposal_methods": ["올바른 이미지를 입력해주세요."],
                        "tips": [],
                    }
                },
                status_code=200,
            )

        # -----------------------------
        # 3️⃣ RAG에서 유사 품목 정보 검색
        # -----------------------------
        db_embeddings = load_embedding_data(EMBED_PATH)
        item_vecs = get_text_embeddings(items)

        rag_contexts = []
        for item, vec in zip(items, item_vecs):
            best_sim = -1
            best_info = None
            for entry in db_embeddings:
                sim = cosine_similarity(vec, entry["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_info = entry
            if best_info:
                rag_contexts.append(
                    f"- 유사 품목: {best_info['item_name']} ({best_sim:.3f})\n"
                    f"- 분류: {best_info['disposal_category']}\n"
                    f"- 처리 가이드: {best_info['disposal_method']}"
                )

        rag_context_text = "\n\n".join(rag_contexts)

        # -----------------------------
        # 4️⃣ GPT에게 최종 disposal_methods 생성 요청
        # -----------------------------
        gpt_prompt = (
            PROMPT_RAG_GUIDE
            + "\n\n[현재 이미지에서 추출된 품목]\n"
            + json.dumps(items, ensure_ascii=False)
            + "\n\n[참고할 재활용 데이터]\n"
            + rag_context_text
        )

        response = client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": gpt_prompt}]}
            ],
        )

        disposal_text = response.output_text.strip()

        try:
            disposal_methods = json.loads(disposal_text)
        except json.JSONDecodeError:
            disposal_methods = [
                s.strip() for s in re.split(r"[\n\-•]", disposal_text) if s.strip()
            ]

        # -----------------------------
        # 5️⃣ GPT에게 업사이클링 팁 생성 요청
        # -----------------------------
        tip_prompt = (
            PROMPT_UPCYCLING_TIP
            + "\n\n[품목 리스트]\n"
            + json.dumps(items, ensure_ascii=False)
        )

        tip_response = client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": tip_prompt}]}
            ],
        )

        tip_text = tip_response.output_text.strip()

        try:
            tips = json.loads(tip_text)
        except json.JSONDecodeError:
            tips = [s.strip() for s in re.split(r"[\n\-•]", tip_text) if s.strip()]

        
        # -----------------------------
        # 6️⃣ 품목별 대표 이미지 URL 추가
        # -----------------------------
        IMAGE_DIR = ROOT_DIR / "image"
        IMAGE_MAP = {
            "플라스틱": "Pet.png",
            "페트": "Pet.png",
            "병": "Pet.png",
            "캔": "Can.png",
            "종이": "Paper.png",
            "상자": "Paper.png",
            "박스": "Paper.png",
            "유리": "Glass.png",
            #"스티로폼": "Styrofoam.png",
            #"비닐": "Vinyl.png",
            #"고철": "Metal.png",
            #"철": "Metal.png",
        }

        selected_img = None
        for item in items:
            normalized = re.sub(r"\s+", "", item)  # 공백 제거
            for key, fname in IMAGE_MAP.items():
                if key in normalized:
                    selected_img = fname
                    break
            if selected_img:
                break

        if selected_img:
            # 프론트엔드 렌더링용 URL (FastAPI static mount 기준)
            image_url = f"/static/{selected_img}"
        else:
            image_url = None  # 매칭 실패 시 None

        parsed["image_url"] = image_url

        # -----------------------------
        # 6️⃣ 최종 결과 반환
        # -----------------------------
        parsed["disposal_methods"] = disposal_methods
        parsed["tips"] = tips

        print("[DEBUG] parsed =", parsed)
        return JSONResponse({"result": parsed}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
