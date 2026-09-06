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

/** Parse a GeoTIFF ArrayBuffer, paint class values onto a canvas, and overlay on the map. */
async function parseAndPaintGeoTIFF(arrayBuffer, feature, featureId) {
    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const rasters = await image.readRasters();

    const probabilityData = rasters[0];
    const width = image.getWidth();
    const height = image.getHeight();

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < probabilityData.length; i++) {
        const val = probabilityData[i];
        const index = i * 4;

        if (val == -1) { // still forested
            imageData.data[index + 0] = 0;
            imageData.data[index + 1] = 128;
            imageData.data[index + 2] = 0;
            imageData.data[index + 3] = 255;
        } else if (val == 1) { // deforested
            imageData.data[index + 0] = 255;
            imageData.data[index + 1] = 71;
            imageData.data[index + 2] = 87;
            imageData.data[index + 3] = 255;
        } else {
            imageData.data[index + 0] = 0;
            imageData.data[index + 1] = 0;
            imageData.data[index + 2] = 0;
            imageData.data[index + 3] = 0;
        }
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
