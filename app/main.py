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


models = {}

TEST_YEAR = 2024

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


TROPIC_LAT = 23.5
BOREAL_LAT = 50.0 
THRESHOLD = 0.5

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

# Endpoint 1: Serves dynamic spatial features
@app.get("/api/locations")
async def get_locations():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Central Park", "description": "New York City"},
                "geometry": {"type": "Point", "coordinates": [-73.9654, 40.7829]}
            },
            {
                "type": "Feature",
                "properties": {"name": "Golden Gate Park", "description": "San Francisco"},
                "geometry": {"type": "Point", "coordinates": [-122.4862, 37.7694]}
            }
        ]
    }

@app.post("/api/predict")
async def predict(polygon: pydantic_models.GeoJSONFeature):
    polygon = shape(polygon.geometry)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    centroid = polygon.centroid
    data, mask, dimensions = get_data_bbox(polygon, TEST_YEAR)
    if data is None:
        return None
    height, width = dimensions
    
    if -TROPIC_LAT < centroid.y < TROPIC_LAT:
        # tropical
        active_model=models['tropical']
    elif TROPIC_LAT < centroid.y < BOREAL_LAT:
        # temperate north
        active_model=models['temperate']
    elif -90 < centroid.y < -TROPIC_LAT:
        # temperate south
        active_model=models['temperate']
    elif BOREAL_LAT < centroid.y < 90:
        # boreal
        active_model=models['boreal']

    data_masked = data[mask]
    
    # 2. Run Inference ONLY on the valid forests
    raw_probs = active_model.predict_proba(data_masked)[:, 1]
    if len(raw_probs) > 0 and raw_probs.max() > raw_probs.min(): # rescale to be between 0 and 1
        valid_probs = (raw_probs - raw_probs.min()) / (raw_probs.max() - raw_probs.min())
    else:
        valid_probs = raw_probs
    # print(list(raw_probs))
    # print(list(valid_probs))
    predictions = np.where(raw_probs > THRESHOLD, 1, -1)

    # 3. Create the blank geographic canvas
    total_pixels = height * width
    flat_output = np.full(total_pixels, 0, dtype=np.float32)
    
    # 4. The Magic Spatial Injection
    # We use the exact same valid_mask to inject the predictions back into 
    # their precise geographic coordinates in the 1D line.
    flat_output[mask] = predictions
    
    # 5. Fold the map back into 2D for Rasterio
    final_2d_map = flat_output.reshape(dimensions)
    
    # Phase 4: Rasterization with Rasterio
    # 1. Calculate Affine Transform
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, height, width)

    mask = geometry_mask(
        [polygon],
        out_shape=dimensions,
        transform=transform,
        invert=True # 'True' means pixels INSIDE the polygon get a True boolean
    )

    final_2d_map[~mask] = 0
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
