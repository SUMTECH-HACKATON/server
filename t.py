# ...existing code...
import json
import os
import time
from typing import List

from dotenv import load_dotenv

# requires: pip install openai python-dotenv
from openai import OpenAI

# 명시적으로 .env 경로 로드 (없다면 그냥 무시)
load_dotenv(r"d:\workspace\cleanbin\.env")

# 설정
INPUT_PATH = r"d:\workspace\cleanbin\rag.json"
OUTPUT_PATH = r"d:\workspace\cleanbin\embedding.json"
MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
RETRY_DELAY = 2.0
MAX_RETRIES = 5

def load_items(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    guides = data.get("recycling_guidelines", [])
    return guides

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def get_embeddings(client: OpenAI, texts: List[str]) -> List[List[float]]:
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = client.embeddings.create(model=MODEL, input=texts)
            # resp.data는 객체 리스트이며 각 항목에 .embedding이 있음
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY * attempt)

def main():
    # 먼저 .env에서 로드했으므로 os.getenv 로 가져옴
    api_key = os.environ.get("OPENAI_API_KEY2") or os.getenv("OPENAI_API_KEY2")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY2가 설정되어 있지 않습니다. .env를 확인하거나 PowerShell에서 $env:OPENAI_API_KEY2='YOUR_KEY' 또는 setx OPENAI_API_KEY2 'YOUR_KEY'로 설정하세요.")
    client = OpenAI(api_key=api_key)

    items = load_items(INPUT_PATH)
    if not isinstance(items, list):
        raise RuntimeError("recycling_guidelines가 리스트가 아닙니다.")

    output_records = []
    texts = [str(item.get("item_name", "")).strip() for item in items]

    for i, batch_idxs in enumerate(chunks(list(range(len(texts))), BATCH_SIZE)):
        batch_texts = [texts[idx] for idx in batch_idxs]
        embeddings = get_embeddings(client, batch_texts)
        for local_idx, idx in enumerate(batch_idxs):
            rec = {
                "index": idx,
                "item_name": items[idx].get("item_name"),
                "disposal_category": items[idx].get("disposal_category"),
                "disposal_method": items[idx].get("disposal_method"),
                "embedding": embeddings[local_idx],
            }
            output_records.append(rec)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"embeddings": output_records, "model": MODEL}, f, ensure_ascii=False, indent=2)

    print(f"완료: {len(output_records)}개 임베딩 -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
# ...existing code...