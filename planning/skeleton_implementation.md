# ForestWatch V1.0: Walking Skeleton Implementation Plan

## Objective
Establish an end-to-end data pipeline from the frontend map to the backend server and back, completely bypassing the complex machine learning and Earth Engine API steps. The goal is to prove that the routing, coordinate geometry, and image rasterization plumbing works. 

**Success State:** A user draws a polygon on the map, and a blurry GeoTIFF image of random "dummy" pixels appears exactly within that bounding box.

---

## Phase 1: Frontend Shell (Web Map & Draw Tools)
**Goal:** Capture user input as spatial data and send it to the backend.
**Tech:** MapLibre GL JS.

### Tasks:
1. **Initialize Map:** Render a basic interactive web map centered on a default location.
2. **Implement Draw Tool:** Add a polygon drawing control to the map UI.
3. **Capture GeoJSON:** When the user finishes drawing, extract the polygon's coordinates into a standard GeoJSON feature object.
4. **API Call:** Create an asynchronous function to send the GeoJSON via a `POST` request to the backend endpoint (`http://localhost:8000/api/predict`).
5. **Render Response:** Configure the map to accept the returned image blob (GeoTIFF/PNG) and overlay it accurately onto the map using the original polygon's bounding coordinates.

---

## Phase 2: Backend Shell (FastAPI & Geo-Routing)
**Goal:** Receive the spatial payload, calculate its bounds, and prepare it for processing.
**Recommended Tech:** FastAPI, Uvicorn, Pydantic, Shapely.

### Tasks:
1. **Server Setup:** Initialize a FastAPI app and configure CORS middleware to allow requests from the frontend development server.
2. **Data Validation:** Create a Pydantic model to validate the incoming GeoJSON payload.
3. **Endpoint Creation:** Build the `POST /detect` endpoint.
4. **Spatial Math (Shapely):** Parse the incoming GeoJSON into a `Shapely` polygon. Extract the polygon's bounding box (`min_lon`, `min_lat`, `max_lon`, `max_lat`).

---

## Phase 3: The "Dummy" Math (Bypassing ML)
**Goal:** Simulate the output of the XGBoost model without making any external API calls or loading real data.
**Recommended Tech:** NumPy.

### Tasks:
1. **Generate Matrix:** Create a small, 2D NumPy array (e.g., 10x10 or 20x20). 
2. **Populate Probabilities:** Fill the array with random floating-point numbers between 0.0 and 1.0. This perfectly mimics the eventual probability array output by `xgboost.predict()`.

---

## Phase 4: Rasterization (The Return Trip)
**Goal:** Stitch the 2D NumPy array of dummy probabilities back into a geographic coordinate system and send it to the browser.
**Recommended Tech:** Rasterio, io (in-memory buffers).

### Tasks:
1. **Calculate Affine Transform:** Use `rasterio.transform.from_bounds()` using the bounding box extracted in Phase 2 and the dimensions of the dummy NumPy array (10x10). This step anchors the matrix to the physical Earth.
2. **Write to Buffer:** Create an in-memory byte buffer (`io.BytesIO()`). Use `rasterio` to write the NumPy array into this buffer as a single-band GeoTIFF file. Assign it a standard Coordinate Reference System (CRS), typically EPSG:4326.
3. **Return Response:** Return the byte buffer directly from the FastAPI endpoint as an HTTP `Response` with the media type `image/tiff` (or convert to PNG if the frontend library prefers).

---

## Acceptance Criteria Check
Before moving on to the real machine learning integration, verify the following:
* [ ] The frontend successfully POSTs a valid GeoJSON object.
* [ ] The backend successfully parses the geographic bounds.
* [ ] The backend returns an image file without crashing.
* [ ] **Crucial:** The returned image aligns *perfectly* over the polygon the user drew on the map, without stretching into the ocean or appearing on the wrong continent.