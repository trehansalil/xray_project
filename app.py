from xray.pipeline.train_pipeline import TrainingPipeline
from fastapi import FastAPI
import uvicorn
import sys
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from xray.pipeline.prediction_pipeline import PredictionPipeline
from xray.exception import CustomException
from xray.constants import *


text:str = "What is machine learing?"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        train_pipeline = TrainingPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    


@app.post("/predict")
async def predict_route(text):
    try:

        obj = PredictionPipeline()
        text = obj.run_pipeline(text)
        return text
    except Exception as e:
        raise CustomException(e, sys) from e
    



if __name__=="__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
    
# obj = TrainingPipeline()

# obj.run_pipeline()