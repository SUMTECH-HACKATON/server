from fastapi import FastAPI

app = FastAPI(
    title="FastAPI Server on Port 8001",
    description="Simple example FastAPI app running on port 8001",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on port 8001!"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

# 이 파일은 uvicorn으로 실행:
# uvicorn main:app --host 0.0.0.0 --port 8001 --reload