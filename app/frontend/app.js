// Initialize MapLibre Map
const map = new maplibregl.Map({
    container: 'map',
    style: 'https://tiles.openfreemap.org/styles/liberty',
    center: [-95.7129, 37.0902],
    zoom: 3
});

// manually set the prefixes of the MapBox class to connect to maplibre instead
MapboxDraw.constants.classes.CONTROL_BASE  = 'maplibregl-ctrl';
MapboxDraw.constants.classes.CONTROL_PREFIX = 'maplibregl-ctrl-';
MapboxDraw.constants.classes.CONTROL_GROUP = 'maplibregl-ctrl-group';

// Inline equivalent of @mapbox/mapbox-gl-draw-static-mode (no npm/bundler needed)
const StaticMode = {
    onSetup: function () {
        this.setActionableState(); // disables trash/combine/uncombine while static
        return {};
    },
    toDisplayFeatures: function (state, geojson, display) {
        display(geojson);
    }
};

const modes = MapboxDraw.modes;
modes.static = StaticMode;

// Initialize Mapbox Draw
const draw = new MapboxDraw({
    displayControlsDefault: false,
    modes: modes,
    controls: {
        polygon: true,
        trash: true
    }
});

// Add the draw control to the map
map.addControl(draw, 'top-left');

map.on('load', () => {
    map.getCanvas().classList.add('mapboxgl-canvas');
    map.getContainer().classList.add('mapboxgl-map');
});

// Listen for when a user finishes drawing a polygon.
// Defer predict/lock: calling changeMode inside draw.create's sync stack
// recurses with Draw's own post-create transition (max call stack exceeded).
map.on('draw.create', (e) => {
    const featureId = e.features[0].id;
    console.log('Created Feature ID:', featureId);
    setTimeout(() => sendPolygonToBackend(featureId), 0);
});
map.on('draw.update', (e) => {
    const featureId = e.features[0].id;
    console.log('Updated Feature ID:', featureId);
    setTimeout(() => sendPolygonToBackend(featureId), 0);
});

// Listen for deletion to clean up the overlay
map.on('draw.delete', (e) => {
    const featureId = e.features[0].id;
    console.log('Deleted Feature ID:', featureId);
    geotiffCache.delete(featureId);
    if (map.getLayer('prediction-overlay-layer-' + featureId)) map.removeLayer('prediction-overlay-layer-' + featureId);
    if (map.getSource('prediction-overlay-' + featureId)) map.removeSource('prediction-overlay-' + featureId);
});

// Prevent re-entrancy: changeMode / render can re-fire draw.update and loop /api/predict
let isPredicting = false;
let ignoreDrawEvents = false;

function lockDraw() {
    if (draw.getMode() !== 'static') {
        draw.changeMode('static');
    }
    // StaticMode does not disable the toolbar — block polygon/trash clicks during predict
    map.getContainer().classList.add('draw-locked');
}

function unlockDraw() {
    map.getContainer().classList.remove('draw-locked');
    draw.changeMode('simple_select');
}

// If a mode change slips through (e.g. before lock CSS applies), snap back outside Draw's stack
map.on('draw.modechange', (e) => {
    if (isPredicting && e.mode !== 'static') {
        setTimeout(() => {
            if (isPredicting) lockDraw();
        }, 0);
    }
});

/** POST a GeoJSON feature to /api/predict and return the GeoTIFF ArrayBuffer. */
async function sendFeatureToBackend(feature) {
    const response = await fetch('api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(feature)
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const arrayBuffer = await response.arrayBuffer();

    // TIFF must begin with 'II' (73) or 'MM' (77). Anything else is likely a JSON/text error.
    const view = new Uint8Array(arrayBuffer);
    if (view[0] !== 73 && view[0] !== 77) {
        const badPayload = new TextDecoder().decode(arrayBuffer);
        console.error("CRITICAL ERROR: Backend did NOT send a TIFF. It sent this text:", badPayload);
        alert("Backend crashed! Check the browser console to read the Python traceback.");
        throw new Error('Backend did not return a GeoTIFF');
    }

    return arrayBuffer;
}

/** RGB for class value: -1 forest, 1 loss; null if nodata/outside. */
function classColor(val, forestRGB, lossRGB) {
    if (val == -1) return forestRGB;
    if (val == 1) return lossRGB;
    return null;
}

/**
 * Blend palette: pred over Hansen truth.
 * Tuned so mixes read as confusion colors (Positive = loss):
 *   TN (no/no) → green, FN (loss/no) → red-brown,
 *   FP (no/loss) → green-brown, TP (loss/loss) → red.
 * Truth weighs more (PRED_ALPHA < 0.5) so disagreements pull toward Hansen's hue.
 */
const PRED_FOREST = [10, 190, 60];
const PRED_LOSS = [250, 30, 30];
const TRUE_FOREST = [0, 150, 50];
const TRUE_LOSS = [200, 35, 15];
const PRED_ALPHA = 0.3;

function blendRGB(predRGB, truthRGB) {
    return [
        Math.round(predRGB[0] * PRED_ALPHA + truthRGB[0] * (1 - PRED_ALPHA)),
        Math.round(predRGB[1] * PRED_ALPHA + truthRGB[1] * (1 - PRED_ALPHA)),
        Math.round(predRGB[2] * PRED_ALPHA + truthRGB[2] * (1 - PRED_ALPHA)),
    ];
}

function rgbCss(rgb) {
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

/** Cache of last GeoTIFF per feature so compare toggle can re-paint without /api/predict. */
const geotiffCache = new Map(); // featureId -> { arrayBuffer, feature }

/** When false, Hansen band is ignored (trueData treated as null) on paint. */
let compareHansen = true;

/** Parse a GeoTIFF ArrayBuffer, paint class values onto a canvas, and overlay on the map.
 *  Band 0 = prediction; band 1 (if present) = Hansen truth, blended underneath.
 *  Hansen is only blended when compareHansen is on.
 *  Parses a copy of the buffer so the cache entry stays valid for re-paints. */
async function parseAndPaintGeoTIFF(arrayBuffer, feature, featureId) {
    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer.slice(0));
    const image = await tiff.getImage();
    const rasters = await image.readRasters();

    const probabilityData = rasters[0];
    const trueData = (compareHansen && rasters.length > 1) ? rasters[1] : null;
    const width = image.getWidth();
    const height = image.getHeight();

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < probabilityData.length; i++) {
        const pred = probabilityData[i];
        const index = i * 4;
        const predRGB = classColor(pred, PRED_FOREST, PRED_LOSS);

        let r = 0, g = 0, b = 0, a = 0;

        if (trueData) {
            const truthRGB = classColor(trueData[i], TRUE_FOREST, TRUE_LOSS);
            if (truthRGB && predRGB) {
                [r, g, b] = blendRGB(predRGB, truthRGB);
                a = 255;
            } else if (predRGB) {
                [r, g, b] = predRGB;
                a = 255;
            } else if (truthRGB) {
                [r, g, b] = truthRGB;
                a = 255;
            }
        } else if (predRGB) {
            [r, g, b] = predRGB;
            a = 255;
        }

        imageData.data[index + 0] = r;
        imageData.data[index + 1] = g;
        imageData.data[index + 2] = b;
        imageData.data[index + 3] = a;
    }

    ctx.putImageData(imageData, 0, 0);
    const dataUrl = canvas.toDataURL('image/png');
    const coordinates = calculateBoundingBox(feature.geometry.coordinates[0]);
    renderImageOverlay(dataUrl, coordinates, featureId);
}

/** Orchestrates predict + paint for a drawn feature (used by draw.create / draw.update). */
async function sendPolygonToBackend(featureId) {
    if (ignoreDrawEvents) return;
    if (isPredicting) return;
    isPredicting = true;
    // Safe here: we are outside Draw's create/update event stack (see setTimeout above)
    lockDraw();

    try {
        const latestFeature = draw.get(featureId);
        if (!latestFeature) return;

        const arrayBuffer = await sendFeatureToBackend(latestFeature);
        geotiffCache.set(featureId, { arrayBuffer: arrayBuffer.slice(0), feature: latestFeature });
        await parseAndPaintGeoTIFF(arrayBuffer, latestFeature, featureId);
    } catch (error) {
        console.error("Failed to process polygon:", error);
        draw.delete(featureId);
    } finally {
        // ignoreDrawEvents blocks any draw.update fired by the mode restore.
        ignoreDrawEvents = true;
        isPredicting = false;
        try {
            unlockDraw();
        } catch (err) {
            console.error('Failed to restore draw mode:', err);
        }
        ignoreDrawEvents = false;
    }
}

// Utility: Calculate min/max lat/lng to get the 4 corners for the image source
function calculateBoundingBox(polygonCoords) {
    let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;

    polygonCoords.forEach(coord => {
        if (coord[0] < minLng) minLng = coord[0];
        if (coord[1] < minLat) minLat = coord[1];
        if (coord[0] > maxLng) maxLng = coord[0];
        if (coord[1] > maxLat) maxLat = coord[1];
    });

    // MapLibre expects image source coordinates in a specific order:
    // [top-left, top-right, bottom-right, bottom-left]
    return [
        [minLng, maxLat], // Top-Left
        [maxLng, maxLat], // Top-Right
        [maxLng, minLat], // Bottom-Right
        [minLng, minLat]  // Bottom-Left
    ];
}

// Utility: Add or update the map layer with the new image
function renderImageOverlay(imageUrl, coordinates, featureId) {
    // If the source already exists, update it rather than duplicating
    if (map.getSource('prediction-overlay-' + featureId)) {
        map.getSource('prediction-overlay-' + featureId).updateImage({ url: imageUrl, coordinates: coordinates });
    } else {
        map.addSource('prediction-overlay-' + featureId, {
            type: 'image',
            url: imageUrl,
            coordinates: coordinates
        });

        map.addLayer({
            id: 'prediction-overlay-layer-' + featureId,
            type: 'raster',
            source: 'prediction-overlay-' + featureId,
            paint: {
                'raster-opacity': 0.7,
                'raster-fade-duration': 0
            }
        });
    }
}



/** Selected year for prediction / map overlays (driven by the year-picker widget). */
let selectedYear = 2024;

/** Re-paint every cached overlay with the current compareHansen setting (no backend). */
async function repaintCachedOverlays() {
    for (const [featureId, cached] of geotiffCache) {
        await parseAndPaintGeoTIFF(cached.arrayBuffer, cached.feature, featureId);
    }
}

async function addWidgets() {
    const response = await fetch('widgets.html');
    if (!response.ok) {
        throw new Error(`Failed to load widgets.html (${response.status})`);
    }
    const content = await response.text();

    // Split widgets.html into separate year-picker, compare-toggle, and legend roots
    const template = document.createElement('template');
    template.innerHTML = content.trim();
    const yearEl = template.content.querySelector('.year-picker');
    const compareEl = template.content.querySelector('.compare-toggle');
    const legendEl = template.content.querySelector('.confusion-legend');
    if (!yearEl || !compareEl || !legendEl) {
        throw new Error('widgets.html missing .year-picker, .compare-toggle, or .confusion-legend');
    }

    const yearPicker = new HtmlWidget({
        content: yearEl.outerHTML,
        position: 'top-right'
    });
    map.addControl(yearPicker);

    const compareWidget = new HtmlWidget({
        content: compareEl.outerHTML,
        position: 'top-right'
    });
    map.addControl(compareWidget);

    const legendWidget = new HtmlWidget({
        content: legendEl.outerHTML,
        position: 'top-right'
    });
    map.addControl(legendWidget);

    // Legend swatches = same blends used when painting (Positive = loss)
    const legendSwatches = {
        tp: blendRGB(PRED_LOSS, TRUE_LOSS),
        tn: blendRGB(PRED_FOREST, TRUE_FOREST),
        fp: blendRGB(PRED_LOSS, TRUE_FOREST),
        fn: blendRGB(PRED_FOREST, TRUE_LOSS),
    };
    for (const [key, rgb] of Object.entries(legendSwatches)) {
        const swatch = document.querySelector(`.confusion-legend__swatch--${key}`);
        if (swatch) swatch.style.background = rgbCss(rgb);
    }

    const slider = document.getElementById('year-slider');
    const valueOut = document.getElementById('year-value');
    if (!slider || !valueOut) return;

    const syncYear = () => {
        selectedYear = Number(slider.value);
        valueOut.textContent = String(selectedYear);
        console.log('Selected year:', selectedYear);
    };

    const updateYear = async () => {
        selectedYear = Number(slider.value);
        const response = await fetch('api/update-year', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ year: selectedYear })
        });
        if (!response.ok) {
            throw new Error(`Failed to update year (${response.status})`);
        }
        const data = await response.json();
        const currentFeatures = draw.getAll();
        console.log('Current features:', currentFeatures);
        currentFeatures.features.forEach((feature) => {
            const featureId = feature.id;
            sendPolygonToBackend(featureId);
        });
    };

    slider.addEventListener('input', syncYear);
    syncYear();
    slider.addEventListener('change', updateYear);
    // Ensure backend config.TEST_YEAR matches the default slider year on load
    await updateYear();

    const compareCheckbox = document.getElementById('compare-hansen');
    const legend = document.getElementById('confusion-legend');
    if (!compareCheckbox) return;

    const syncLegendVisibility = () => {
        if (!legend) return;
        legend.classList.toggle('is-hidden', !compareCheckbox.checked);
    };
    syncLegendVisibility();

    compareCheckbox.addEventListener('change', () => {
        compareHansen = compareCheckbox.checked;
        syncLegendVisibility();
        console.log('Compare Hansen:', compareHansen);
        repaintCachedOverlays().catch((err) => {
            console.error('Failed to re-paint overlays after compare toggle:', err);
        });
    });
}

addWidgets().catch((err) => {
    console.error('Map widgets failed to load:', err);
});