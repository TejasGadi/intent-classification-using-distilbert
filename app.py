from fastapi import FastAPI
import uvicorn

import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

from fastapi.responses import Response
from src.textClassifier.pipeline.prediction_pipeline import PredictionPipeline

text: str = "What is Text Summarization?"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train():
    try:
        os.system("python main.py")
        return Response("Training Successful")
    except Exception as e:
        return Response(f"Error: {e}")

@app.get("/predict")
async def predict_route(text: str):
    try:
        pred_pipeline = PredictionPipeline()
        summary = pred_pipeline.predict(text)
        return summary
    except Exception as e:
        raise e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)