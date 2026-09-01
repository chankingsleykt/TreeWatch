from typing import Annotated
from fastapi import FastAPI, Body, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pydantic_models
from shapely.geometry import shape
import numpy as np
from PIL import Image
import io
from contextlib import asynccontextmanager
from xgboost import XGBClassifier
from rasterio.transform import from_bounds
from rasterio.io import MemoryFile
from rasterio.features import geometry_mask
from ee_connection import get_data_bbox
from config import TROPIC_LAT, BOREAL_LAT, THRESHOLD
from prediction_helpers import route_model_and_predict

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

@app.post("/api/predict")
async def predict(polygon: pydantic_models.GeoJSONFeature):
    polygon = shape(polygon.geometry)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    centroid = polygon.centroid
    result = get_data_bbox(polygon)
    if result["error"] is not None:
        return result['error']
    data = result["data"]
    mask = result["mask"]
    height, width = result["coords"]
    transform = result["transform"]
    data_masked = data[mask]
    predictions = route_model_and_predict(centroid, data_masked, models)

    # create blank geographic canvas
    total_pixels = height * width
    flat_output = np.full(total_pixels, 0, dtype=np.float32)
    
    # We use the exact same valid_mask to inject the predictions back into their precise geographic coordinates in the 1D line.
    flat_output[mask] = predictions
    
    # fold the map back into 2D for Rasterio
    final_2d_map = flat_output.reshape((height, width))

    polygon_mask = geometry_mask(
        [polygon],
        out_shape=(height, width),
        transform=transform,
        invert=True # 'True' means pixels INSIDE the polygon get a True boolean
    )

    final_2d_map[~polygon_mask] = 0
    print(final_2d_map)
    # 2. Write to Buffer
    # MemoryFile acts as a virtual filesystem for rasterio
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=1,                  # Single band
            dtype=final_2d_map.dtype,
            crs='EPSG:4326',          # Standard WGS84 coordinates
            transform=transform,
            nodata=0
        ) as dataset:
            # Write the numpy array to the first band
            dataset.write(final_2d_map, 1)

        # Extract the raw bytes from the MemoryFile
        tiff_bytes = memfile.read()

    # 3. Return Response
    return Response(content=tiff_bytes, media_type="image/tiff")

# Endpoint 2: Serves the MapLibre Map UI
@app.get("/", response_class=HTMLResponse)
async def get_map():
    with open("frontend/index.html", "r") as f:
        return f.read()
