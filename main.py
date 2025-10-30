from fastapi import FastAPI
from vision_analyzer import router as vision_router  # ✅ 모듈 import

app = FastAPI(
    title="FastAPI Server on Port 8001",
    description="Simple example FastAPI app running on port 8001",
    version="1.0.0"
)

# 기본 라우트
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on port 8001!"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

# ✅ vision_analyzer 라우터 등록
app.include_router(vision_router)

# 실행:
# uvicorn main:app --host 0.0.0.0 --port 8001 --reload
