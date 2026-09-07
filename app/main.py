from typing import Annotated
from fastapi import FastAPI, Body, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pydantic_models
from shapely.geometry import shape
import numpy as np
from PIL import Image
import io
from contextlib import asynccontextmanager
from xgboost import XGBClassifier
from rasterio.io import MemoryFile
from ee_connection import get_data_bbox
import config
from prediction_helpers import route_model_and_predict, values_to_raster

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP SEQUENCE ---
    print("Loading models into memory...")
    
    # Initialize empty XGBoost Classifier objects
    models['tropical'] = XGBClassifier()
    models['temperate'] = XGBClassifier()
    models['boreal'] = XGBClassifier()
    
    # Load the JSON weights into the objects
    models['tropical'].load_model("ml_models/xgb_tropical.json")
    models['temperate'].load_model("ml_models/xgb_temperate.json")
    models['boreal'].load_model("ml_models/xgb_boreal.json")
    
    yield # The server is now live and accepting requests
    
    # --- SHUTDOWN SEQUENCE ---
    print("Clearing models from memory...")
    models.clear()


app = FastAPI(title="ForestWatch", lifespan=lifespan)


origins = [
    "http://localhost:3000",    # React default port
    "http://localhost:5173",    # Vite default port
    "http://127.0.0.1:5173",
    # Add your production domain here later
]

# Enable CORS for the frontend map to access the backend data
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/toggle-compare")
async def toggle_compare(body: dict = Body(...)):
    compare = body.get('compare')
    print(f"Toggling compare to {compare}")
    config.COMPARE = compare
    return {"message": f"Compare toggled to {compare}"}

@app.post("/api/predict")
async def predict(body: pydantic_models.PredictRequest):
    polygon = shape(body.feature.geometry)
    year = body.year
    print(f"Predict for year {year}")

    centroid = polygon.centroid
    result = get_data_bbox(polygon, year)
    if result["error"] is not None:
        return result['error']
    data = result["data"]
    mask = result["mask"]
    height, width = result["coords"]
    transform = result["transform"]
    data_masked = data[mask]
    features = data_masked[config.BANDS_IN_ORDER]
    predictions = route_model_and_predict(centroid, features, models)
    prediction_map = values_to_raster(predictions, mask, height, width, polygon, transform)
    print(prediction_map)

    include_hansen = 'loss_true' in data_masked.columns
    hansen_map = None
    if include_hansen:
        # Match prediction encoding: 1 = loss, -1 = no loss (within valid forest mask)
        hansen_truth = np.where(data_masked['loss_true'].to_numpy(), 1, -1).astype(np.float32)
        hansen_map = values_to_raster(hansen_truth, mask, height, width, polygon, transform)

    band_count = 2 if include_hansen else 1
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=band_count,
            dtype=prediction_map.dtype,
            crs='EPSG:4326',
            transform=transform,
            nodata=0
        ) as dataset:
            dataset.write(prediction_map, 1)
            if include_hansen:
                dataset.write(hansen_map, 2)

        tiff_bytes = memfile.read()

    return Response(content=tiff_bytes, media_type="image/tiff")




@app.get("/", response_class=HTMLResponse)
async def get_map():
    with open("frontend/index.html", "r") as f:
        return f.read()

app.mount("/", StaticFiles(directory="frontend"), name="frontend")
