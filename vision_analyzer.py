import base64
import json
import os
import re
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
    """
    다음 재활용품 사진이 무슨 물건인지 json으로 으로 명사로 반환해줘.
    
    출력 형식 예시는 다음과 같아야 해:
    
    {"items": ["플라스틱병", "종이상자", "유리병", "종이컵"],
    "materials": ["비닐류", "플라스틱", "종이", "유리", "금속", "고철","스티로폼"],
    "details": ["뚜껑이 있는 플라스틱병", "박스 형태의 종이상자", "투명한 유리병", "지저분한 종이컵"]
    }
    
    디테일 같은 부분은 다음을 챙겨줘
    1. 안에 내용물이 있는지 없는지
    2. 라벨이 붙어있는지 아닌지
    3. 뚜껑이 있는지 없는지
    4. 찌그러져 있는지 아닌지
    5. 깨져있는지 아닌지
    6. 기타 시각적으로 구분되는 특징들
    7. 더러운지 아닌지
    반드시 JSON 형식으로만 응답해줘. 다른 설명이나 문장은 포함하지마.
    """
)

PROMPT2=(
    """
        다음 재활용품 종류가 리스트로 들어가면 각 재활용품을 재활용하는 방법을 순서대로 디테일하게 반환해줘 텍스트가 들어간 리스트가 필요해.
        예를 들어 
        ["플라스틱병", "종이상자", "유리병"] 가 들어가면
        
        ["#1.플라스틱병은 내용물을 비워주세요 #2. 라벨을 제거해주세요, #3뚜껑을 닫아주세요.  분리수거함에 넣으세요.",
        "종이상자는 내용물을 비우고 납작하게 접은 후, 종이류 수거함에 넣으세요.",
        "유리병은 내용물을 비우고 라벨을 제거한 후, 뚜껑을 닫아 유리병 수거함에 넣으세요."]
        이렇게 반환해줘.
        반드시 리스트 형식으로만 응답해줘. 다른 설명이나 문장은 포함하지마.
        
        그리고 아래는 한국에서 재활용을 하는 방법에 대한 줄글이야. 아래 내용을 참고해서 만들어줘. 
        신문은 종이류로 분류되며 젖지 않게 펴서 묶어 배출하고, 책자나 노트는 스프링과 표지를 제거 후 종이류로 배출한다. 상자류는 테이프를 제거하고 접어서 묶어 종이류로, 깨끗한 종이컵은 헹군 후 종이류로, 오염된 종이컵은 종량제 봉투에 버린다. 우유팩과 주스팩은 세척 후 말리고 다른 재질을 제거해 종이팩 전용 수거함에 넣는다. 페트병은 내용물을 비우고 라벨과 뚜껑을 제거해 플라스틱류로, 플라스틱 용기류는 깨끗이 씻고 부속품을 제거해 플라스틱류로 배출한다. 비닐봉투나 포장재는 이물질을 제거해 비닐류로, 에어캡(뽁뽁이)은 깨끗이 하여 비닐류로 배출한다. 스티로폼 완충재는 세척 후 테이프를 제거해 발포합성수지 수거함에 넣고, 오염된 경우 종량제 봉투로 버린다. 철캔과 알루미늄캔은 세척 후 이물질을 제거해 금속캔 수거함에, 부탄가스와 살충제 용기는 잔류가스를 완전히 제거 후 금속캔류로 배출한다. 고철은 이물질을 제거하고 금속만 분리해 고철류로 배출한다. 음료수병은 헹군 후 유리병류로, 소주·맥주병은 환급 대상이므로 소매점이나 회수기를 통해 반환한다. 깨진 유리는 신문지로 감싸 안전하게 종량제 봉투 또는 불연성폐기물로 처리하며, 내열유리는 불연성폐기물 전용봉투로 배출한다. 폐의류는 폐의류 전용수거함에, 폐식용유는 전용 수거함에 넣는다. 폐형광등과 폐건전지, 폐의약품은 유해폐기물 전용수거함에 넣고, 대형가전제품은 무상 방문수거 서비스를 이용한다. 소형가전은 다량일 경우 함께 수거 요청하며, 가구류는 대형폐기물로 신고 후 스티커를 부착해 배출한다. 음식물류 폐기물은 수분을 제거해 전용 용기에 배출하고, 알껍질·뼈·씨 등은 음식물류 폐기물이 아니므로 종량제 봉투에 버린다. 감열지나 코팅 종이, 광고지, 사진은 재활용 불가이므로 종량제 봉투로 배출한다. 아이스팩은 겉과 속을 분리할 수 있으면 비닐류로, 아니면 종량제 봉투로 배출한다. 옷걸이, 칫솔, 알약 포장재 등 복합재질은 분리가 어려우면 종량제 봉투에 버린다. 고무장갑, 백열전구, 도자기, 타일 등은 재활용 불가 품목으로 각각 종량제 봉투나 불연성 폐기물 전용봉투에 담는다. 다 쓴 라이터는 내용물을 제거 후 종량제 봉투에 버리고, 우산은 뼈대와 천을 분리 후 각각 재질에 맞게 배출한다. 나무젓가락·도마·국자 등 나무류는 일반쓰레기, 낫·못·나사 등은 고철류로 분류한다. 플라스틱 도마·빨대·분무기 등은 세척 후 플라스틱류로 배출한다. 콘센트, 전선류, 마우스패드, 마스크, 면봉, 볼펜, 연필 등은 재활용 불가로 종량제 봉투에 버린다. 골프공, 자석, 종이기저귀, 커피 찌꺼기, 차 찌꺼기, 양초, 비디오테이프, CD/DVD 등도 일반쓰레기로 처리한다. 낚싯대, 자전거, 카펫, 텐트, 가구류 등 부피가 큰 품목은 대형폐기물로 신고 후 스티커를 부착한다. 자동차 부품, 타이어, 기름은 전문처리시설로 보내야 한다. 애완동물 용변 시트는 종량제 봉투로, 음식캔은 세척 후 금속캔으로 배출한다. 폐인트통은 내용물이 없으면 금속캔류, 남아 있으면 유해폐기물로 처리한다. 젖병은 몸체와 젖꼭지를 분리해 각각 재질에 맞게 배출한다. 면도칼, 조각칼 등 날카로운 물품은 종이에 감싸 안전하게 종량제 봉투로 버린다.

        
    """
)

# FastAPI Router 생성
router = APIRouter(prefix="/vision", tags=["GPT Vision Analyzer"])


def image_bytes_to_data_url(b: bytes, ctype: str = "image/png") -> str:
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{ctype};base64,{b64}"

def call_gpt(data_url: str, model: str = DEFAULT_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT2},
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()

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

        # ✅ 크기 제한
        w, h = img.size
        scale = max(w, h) / float(MAX_SIDE)
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

        # ✅ PNG/JPG 형식 유지
        fmt = "PNG" if file.filename.lower().endswith(".png") else "JPEG"
        ctype = "image/png" if fmt == "PNG" else "image/jpeg"

        bio = BytesIO()
        img.save(bio, format=fmt, quality=90 if fmt == "JPEG" else None)

        data_url = image_bytes_to_data_url(bio.getvalue(), ctype=ctype)
        
        result_text = call_gpt_vision(data_url, model=model)

        # ✅ JSON 추출 시도
        try:
            match = re.search(r"\{[\s\S]*\}", result_text)
            if not match:
                raise ValueError("No JSON object found in GPT response")

            parsed = json.loads(match.group(0))

            # ✅ items 필드 검증
            items = parsed.get("items", [])
            if not isinstance(items, list):
                raise ValueError("items 필드가 리스트 형태가 아닙니다.")

            # ✅ 만약 items가 비어 있으면 — 사용자에게 안내 메시지 반환
            if not items:
                parsed = {
                    "message": "잘못된 이미지입니다.",
                    "items": [],
                    "disposal_methods": ["올바른 이미지를 입력해주세요."],
                    "description": "올바른 이미지를 입력해주세요.",
                }
                return JSONResponse({"result": parsed}, status_code=200)

            # ✅ GPT에게 재활용 방법 요청
            items_json = json.dumps(items, ensure_ascii=False)
            disposal_response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": PROMPT2 + f"\n\n입력 리스트: {items_json}"}
                        ],
                    }
                ],
            )

            disposal_text = disposal_response.output_text.strip()

            # ✅ disposal 결과 파싱
            match_list = re.search(r"\[.*\]", disposal_text, re.DOTALL)
            if match_list:
                disposal_methods = json.loads(match_list.group(0))
            else:
                disposal_methods = [disposal_text]

            parsed["disposal_methods"] = disposal_methods

        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"JSON parse error: {str(e)}")

        print("[DEBUG] parsed =", parsed)
        return JSONResponse({"result": parsed})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
