# May 2, 2026
* Selected XGBoost as it did best in model_comparison.ipynb
* Evaluated performance on testing data, realized model had high precision but low recall on Amazon and Taiga 2024
* Ran K-S test, confirmed shift in distribution
* Decision: users can manually adjust the probability threshold to account for effects in years
* Realized data leakage in delta calculation methods, where the delta is with the current year rather than current year - 1
    * Fixed data_collection.ipynb, generated new datasets, re-ran model_comparison.ipynb
    * All models now do disastrously worse
* TODO: build region-specific models, hopefully those do well enough, if they still fail, deep learning may be necessary.

# May 23, 2026
* Built the region-specific Amazon model. Still did badly, though there was improvement
* Upon more research, a 30.91 x 30.91 region is about as big as an Olympic-sized swimming pool. My assumption that looking at past spectral indices could "catch deforestation in the act" is wrong because deforesters could clear that region in minutes.
* Decision: pivot to classifying deforestation in real time. While this is essentially recreating the Hansen loss label, the Hansen dataset is static only up to 2025, while this model can be used for future years and dynamically called for specific regions. Essentially a GFW-lite
* TODO: retrain models on unlagged data

# June 21, 2026
* Retrained the models on unlagged data. XGBoost gets 0.99 scores, so I'm thinking that it's too easy to only predict undeforested vs deforested in 2023. I'll try including pixels deforested in previous years in the 0 class and seeing how that goes.

# June 24, 2026
* Tested the model on both data in the most recent year and data from different forests. Model failed to generalize for the very different SE USA forest. The next step is to train three models on each of the three forest types--Tropical, Temperate, Boreal.

# June 25, 2026
* Trained three models for each of the three forest types. Models successfully predicted deforestation in the forests they were trained on, in 2025. While accuracy did decrease on new forests of the same type, f1 scores remain respectable at 0.70. Now development of the app can begin.

# Sep 1, 2026
* Refactored `get_data_bbox` to return a dict (`data`, `mask`, `coords`, `transform`, `error`) instead of a tuple; updated `main.py`, tests, and removed unused import in `prediction_helpers.py`.
* Frontend now removes the MapBox Draw feature when backend crashes to avoid repeated calls
* Hardcoded Hansen `EPSG:4326` / 30 m in `get_data_bbox` and removed all `getInfo()` calls to cut EE round-trip latency
* Added local bbox grid snapping so affine transforms stay aligned with EE exports without `getInfo()`

# Sep 5, 2026
* Moved frontend map logic from inline `<script>` in `index.html` into `frontend/app.js`; mounted `StaticFiles` so FastAPI serves it
* Refactored `sendPolygonToBackend` into `sendFeatureToBackend` (POST + TIFF validation), `parseAndPaintGeoTIFF` (parse/paint/overlay), and orchestrator still used by `draw.create` / `draw.update`
* Added inline Mapbox Draw `StaticMode` (equivalent to `@mapbox/mapbox-gl-draw-static-mode`) and registered it on `MapboxDraw.modes`; lock drawing during `/api/predict` then restore `simple_select`
* Fixed `/api/predict` infinite loop: `isPredicting` guard ignores re-entrant `draw.update` fired by `changeMode`; restore `simple_select` in `finally` before clearing the flag
* Fixed shape still draggable during predict: re-assert `static` after Draw’s post-`create` switch to `simple_select` (`draw.modechange` + `setTimeout(0)`)
* Fixed stack overflow from `draw.modechange`↔`changeMode('static')` recursion by only re-locking when `e.mode !== 'static'`; pass `featureId` into `renderImageOverlay`
* Fixed remaining stack overflow: removed `draw.modechange` re-lock; defer `sendPolygonToBackend` with `setTimeout(0)` so `changeMode('static')` is not called inside Draw’s create/update stack
* Disabled Draw toolbar (polygon/trash) during predict via `#map.draw-locked`; deferred modechange snap-back if draw mode is forced while predicting
* Allow for creating multiple features and maintaining drawn pixels
* Added ability to select a 2001–2026 year range slider and change the year of the data, testing still neede

# Sep 6, 2026
* Fixed pandas `SettingWithCopyWarning` in `get_data_bbox` by copying the band-selected DataFrame before assigning `loss`
* Extracted `values_to_raster` in `prediction_helpers.py`; `/api/predict` now exports a 2-band GeoTIFF (prediction + Hansen ground truth from `data_masked['loss']`)
* `/api/predict` only adds Hansen band when `data_masked` has a `loss` column (omitted for 2026 in `get_data_bbox`)
* `parseAndPaintGeoTIFF` blends Hansen truth (darker green/red) under prediction when band 1 is present
* Call `updateYear()` on year-picker init so `/api/update-year` sets `config.TEST_YEAR` to the default 2024 at startup
* Added Compare Hansen toggle widget; caches GeoTIFF ArrayBuffers per feature and re-paints overlays client-side when toggled (trueData ignored when off; no backend re-call)
* Retuned pred/Hansen blend palette (`PRED_*` / `TRUE_*` / `PRED_ALPHA`) so mixes read as TN green, FN red-brown, FP green-brown, TP red
* Added confusion-matrix legend widget (TP/TN/FP/FN, Positive = loss); swatches driven by the same blend function; hidden when Compare Hansen is off
