import base64
import mimetypes
import os
import time
from io import BytesIO

import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# -------------------- 환경 --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY2")
MODEL = "gpt-4.1-mini"  # Vision 지원 모델

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 가 필요합니다.")

client_ai = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- 설정 --------------------
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
MAX_SIDE = 1280
JPEG_QUALITY = 85

# -------------------- 프롬프트 --------------------
TEXT_PROMPT = (
    "아래 사진은 재활용 쓰레기 또는 생활폐기물의 일부일 수 있습니다.\n"
    "이 사진을 보고 **분리배출 방법**을 한국어로 한 문단(약 200~350자)으로 설명해줘.\n"
    "지켜야 할 규칙:\n"
    "- 항목별로 분리배출 가능한 재질(플라스틱, 종이, 유리, 금속, 비닐 등)을 정확히 구분하고,\n"
    "  재활용이 불가능한 경우 그 이유(오염, 복합재질 등)를 짧게 언급해.\n"
    "- 불확실한 경우에는 '~추정', '~가능성 있음'처럼 완곡하게 표현해.\n"
    "- 불필요한 감정 표현이나 추측(예: 깨끗하다, 더럽다, 오래된 듯)은 쓰지 마.\n"
    "- 예시나 번호를 쓰지 말고 문장형으로 자연스럽게 서술해.\n"
)

# -------------------- 유틸 --------------------
def try_https(u: str) -> str:
    """http:// → https:// 자동 변환"""
    return ("https://" + u[len("http://"):]) if u.startswith("http://") else u

def fetch_to_data_url(url: str) -> str:
    """이미지 URL을 다운로드 → 리사이즈 → base64 data URL로 변환"""
    u = try_https(url)
    headers = {"User-Agent": UA, "Referer": url}
    r = requests.get(u, headers=headers, timeout=12, stream=True, allow_redirects=True)
    r.raise_for_status()

    img = Image.open(BytesIO(r.content)).convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(MAX_SIDE)
    if scale > 1.0:
        img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)

    bio = BytesIO()
    img.save(bio, format="JPEG", quality=JPEG_QUALITY)
    b = bio.getvalue()

    ctype = "image/jpeg"
    guess = mimetypes.guess_type(u)[0]
    if guess and guess.startswith("image/"):
        ctype = guess

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{ctype};base64,{b64}"

def one_paragraph(text: str) -> str:
    """여러 줄 텍스트를 한 문단으로 변환"""
    return " ".join(line.strip() for line in (text or "").splitlines()).strip()

def build_input_for_image(data_url: str):
    """OpenAI Responses API 입력 형식 구성"""
    return [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": TEXT_PROMPT},
            {"type": "input_image", "image_url": data_url},
        ],
    }]

def call_recycle_analyzer(data_url: str) -> str:
    """OpenAI Vision 호출"""
    resp = client_ai.responses.create(
        model=MODEL,
        input=build_input_for_image(data_url),
    )
    return one_paragraph(resp.output_text)

# -------------------- 실행 --------------------
if __name__ == "__main__":
    test_url = input("분석할 쓰레기 이미지 URL을 입력하세요: ").strip()
    if not test_url:
        print("❌ 이미지 URL이 필요합니다.")
        exit(1)

    try:
        data_url = fetch_to_data_url(test_url)
        result = call_recycle_analyzer(data_url)
        print("\n♻️ 재활용 분리수거 안내:")
        print(result)
    except Exception as e:
        print("⚠️ 오류:", e)
