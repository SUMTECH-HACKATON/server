# --------------------------------------------------
# embedding_util.py
# --------------------------------------------------
import json
import os
import time
from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------
# ✅ 환경변수 로드 (.env를 프로젝트 루트에서 자동 탐색)
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent  # cleanbin 폴더 기준
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY2")
if not OPENAI_API_KEY:
    raise RuntimeError(f"환경 변수 OPENAI_API_KEY2가 설정되어 있지 않습니다. (.env: {ENV_PATH})")

# ✅ OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------
# 모델 및 설정값
# --------------------------------------------------
MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
RETRY_DELAY = 2.0
MAX_RETRIES = 5


# --------------------------------------------------
# 내부 유틸 함수
# --------------------------------------------------
def _chunks(lst, n):
    """리스트를 n개씩 나누는 제너레이터."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """OpenAI Embedding API 호출 (재시도 포함)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            print(f"[Retry {attempt}/{MAX_RETRIES}] 임베딩 요청 실패: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY * attempt)


# --------------------------------------------------
# ✅ 공개 함수: 텍스트 임베딩 생성
# --------------------------------------------------
def get_text_embeddings(
    texts: Union[str, List[str]],
    model: str = MODEL
) -> List[List[float]]:
    """
    주어진 텍스트(또는 텍스트 리스트)에 대해 OpenAI 임베딩을 반환합니다.

    Parameters
    ----------
    texts : str | List[str]
        단일 문자열 또는 문자열 리스트.
    model : str, optional
        사용할 임베딩 모델 (기본: text-embedding-3-small)

    Returns
    -------
    List[List[float]]
        각 입력 텍스트에 대한 임베딩 벡터 리스트.

    Example
    -------
        >>> from embedding_util import get_text_embeddings
        >>> vectors = get_text_embeddings(["플라스틱병", "종이컵"])
        >>> len(vectors), len(vectors[0])
        (2, 1536)
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("입력된 텍스트가 비어 있거나 잘못되었습니다.")

    all_embeddings = []
    for batch in _chunks(texts, BATCH_SIZE):
        batch_embeddings = _get_embeddings_batch(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# --------------------------------------------------
# ✅ 파일 단위로 직접 실행 시 (테스트용)
# --------------------------------------------------
if __name__ == "__main__":
    sample_texts = ["플라스틱병", "종이컵", "유리병"]
    embs = get_text_embeddings(sample_texts)
    print(f"{len(embs)}개 임베딩 생성 완료. 첫 벡터 길이 = {len(embs[0])}")
