import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictionRequest, PredictionResponse
from .service import predict

app = FastAPI(title="Uma Probability API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in __import__("os").getenv("CORS_ORIGINS", "http://localhost:8080").split(",")
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo")
def demo():
    return json.loads((Path(__file__).parent.parent / "web" / "demo-race.json").read_text())


@app.get("/api/demo-catalog")
def demo_catalog():
    return json.loads((Path(__file__).parent.parent / "web" / "demo-catalog.json").read_text())


@app.post("/api/predict", response_model=PredictionResponse)
def prediction(request: PredictionRequest):
    return predict(request)
