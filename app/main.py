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
from ee_connection import eliminate_non_forest_and_route_model


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


TROPIC_LAT = 23.5
BOREAL_LAT = 50.0 

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

    eliminate_non_forest_and_route_model(polygon)


    # Phase 3: The "Dummy" Math (Bypassing ML)
    # Generate a 20x20 matrix of random probabilities between 0.0 and 1.0
    probability_matrix = np.random.rand(20, 20).astype(np.float32)
    # print(probability_matrix.shape)


    # Phase 4: Rasterization with Rasterio
    # 1. Calculate Affine Transform
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, 20, 20)

    mask = geometry_mask(
        [polygon],
        out_shape=(20, 20),
        transform=transform,
        invert=True # 'True' means pixels INSIDE the polygon get a True boolean
    )

    probability_matrix[~mask] = np.nan

    # 2. Write to Buffer
    # MemoryFile acts as a virtual filesystem for rasterio
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=20,
            width=20,
            count=1,                  # Single band
            dtype=probability_matrix.dtype,
            crs='EPSG:4326',          # Standard WGS84 coordinates
            transform=transform,
            nodata=np.nan
        ) as dataset:
            # Write the numpy array to the first band
            dataset.write(probability_matrix, 1)

        # Extract the raw bytes from the MemoryFile
        tiff_bytes = memfile.read()

    # 3. Return Response
    return Response(content=tiff_bytes, media_type="image/tiff")

# Endpoint 2: Serves the MapLibre Map UI
@app.get("/", response_class=HTMLResponse)
async def get_map():
    with open("frontend/index.html", "r") as f:
        return f.read()
