# ForestWatch V1.1: Architecture & System Blueprint

## 1. Core Philosophy
ForestWatch is a **Live Change Detector (Nowcasting)** application, not a forecasting tool. It acts as a low-latency Surrogate Model for the Global Forest Watch (Hansen) dataset. The system allows users to draw a polygon on a map and instantly evaluate active deforestation using a hardware-aware, vectorized inference pipeline.

## 2. The Machine Learning Engine
* **The Models:** Three mathematically isolated XGBoost models loaded into server RAM (`Tropical`, `Temperate`, `Boreal`). 
* **The Exclusions:** Subtropical and commercial silviculture regions are explicitly rejected by the routing layer to prevent "poisoning" the Temperate model with rapid-cycle logging data.
* **The Features:** True $\Delta$ (Delta) Architecture. The model evaluates the absolute change between the current state ($t$) and the baseline state ($t-1$) using Landsat 8 optical bands (B4, B5, B6, B7) and spectral indices (NDVI, NBR, NDMI). Annual medians are used to natively filter cloud shadows.
* **Hard Negative Mining:** Pixels deforested in prior years ("Old Dirt") are explicitly labeled as the `0` class during training. This mathematically forces XGBoost to trigger on the *act of changing* ($\Delta$), preventing false positives on pre-existing farms or roads.

## 3. System Constraints & UX Rules
* **Area Limit:** User-drawn polygons are strictly capped at **500 square kilometers** to prevent `Error Code 3` memory crashes from the Earth Engine API.
* **Visual Integrity:** The app renders raw 30-meter pixels. Aggregation grids are strictly forbidden, as majority-voting erases the narrow, sub-pixel signals of illegal logging roads.

## 4. The Execution Pipeline (Step-by-Step)
This is the strict chronological flow of a single user request:

1.  **Trigger:** User draws a polygon on the frontend map and clicks "Analyze".
2.  **Transport 1:** Frontend sends the bounding geometry as a lightweight **GeoJSON** payload to the FastAPI `/detect` endpoint.
3.  **The Orchestrator:** FastAPI calculates the centroid of the GeoJSON and routes the request to the appropriate biome (Tropical, Temperate, Boreal). If the centroid lands in a commercial exclusion zone, FastAPI aborts and returns an HTTP 400 error.
4.  **Vectorized Extraction:** FastAPI sends the GeoJSON to Google Earth Engine. Earth Engine masks out oceans and pixels that were not forests in the baseline year (using historical Hansen `treecover` data), returning a single, flat NumPy array of valid pixels to the backend.
5.  **Inference:** FastAPI passes the NumPy matrix into the active XGBoost model (`xgboost.predict()`), generating a 1D array of probabilities (0.0 to 1.0).
6.  **Rasterization:** FastAPI utilizes `rasterio` to stitch the 1D probability array back into a georeferenced 2D image (Cloud Optimized GeoTIFF).
7.  **Transport 2:** FastAPI serves the GeoTIFF to the frontend.
8.  **Rendering:** The frontend mapping library overlays the GeoTIFF on the map. The GPU shader paints confident pixels red/green and leaves everything else transparent.