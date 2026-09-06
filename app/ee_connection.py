import shapely
import ee
import math
import numpy as np
import pandas as pd
import os
import requests
import io
import sys
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(project_root)
import config
from rasterio.transform import Affine

# Hansen native projection — avoids synchronous EE getInfo() round trips
HANSEN_CRS = "EPSG:4326"
HANSEN_SCALE_M = 30

load_dotenv()
project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID") # insert your id here

ee.Authenticate(force=False)
ee.Initialize(project=project_id)


def snap_bbox_to_grid(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    scale_m: float = HANSEN_SCALE_M,
) -> tuple[float, float, float, float]:
    """Snap a WGS84 bbox outward to the export pixel grid at ``scale_m`` meters."""
    center_lat = (min_lat + max_lat) / 2
    lat_rad = math.radians(center_lat)
    pixel_width = scale_m / (111320.0 * math.cos(lat_rad))
    pixel_height = scale_m / 111320.0

    west = math.floor(min_lon / pixel_width) * pixel_width
    south = math.floor(min_lat / pixel_height) * pixel_height
    east = math.ceil(max_lon / pixel_width) * pixel_width
    north = math.ceil(max_lat / pixel_height) * pixel_height
    return west, south, east, north


def get_data_bbox(polygon: shapely.Polygon) -> dict:
    """
    Creates a GeoJSON of the bounding box containing the polygon, extracts the landsat image bands, filters by Hansen treecover and lossyear data,
    and returns a pandas DataFrame with the landsat image data, and nan where treecover < 50 or if it's been deforested before TEST_YEAR
    
    Args:
        polygon: A Shapely Polygon geometry
        
    Returns:
        dict with keys:
        "data": A pandas DataFrame with the landsat image bands and the true Hansen loss values, or None on error
        "mask": A numpy array with the same length as data, True only if treecover > 50 and lossyear = 0 or lossyear > TEST_YEAR, or None on error
        "coords": (height, width) tuple of the bounding box grid dimensions, or None on error
        "transform": Affine transform for reconverting each row in the dataframe to a pixel in the polygon, or None on error
        "error": "polygon too large" if the request exceeds Earth Engine size limits, the original error message for other failures, or None on success
    """

    # Snap bbox to the 30 m grid so the affine transform matches EE's export footprint
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    west, south, east, north = snap_bbox_to_grid(min_lon, min_lat, max_lon, max_lat)
    bbox_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south]
        ]]
    }
    bbox_geometry = ee.Geometry(bbox_geojson)
    print(f"TEST_YEAR: {config.TEST_YEAR}")
    # extract landsat bands
    landsat_image_lag = process_yearly_landsat(config.TEST_YEAR-1, 1, 1, config.TEST_YEAR, 1, 1)
    landsat_image_current = process_yearly_landsat(config.TEST_YEAR, 1, 1, config.TEST_YEAR, 1, 1)
    landsat_image = landsat_image_current.addBands(landsat_image_lag)

    # extract hansen bands for filtering
    hansen = ee.Image('UMD/hansen/global_forest_change_2025_v1_13')

    # reshape the landsat image to hansen level
    landsat_image = landsat_image.resample('bilinear').reproject(
        crs=HANSEN_CRS, scale=HANSEN_SCALE_M
    )

    combined_image = landsat_image.addBands([
        hansen.select('treecover2000'),
        hansen.select('lossyear')
    ])

    # request the data from Earth Engine as NPY
    try:
        url = combined_image.getDownloadURL({
            'region': bbox_geometry,
            'scale': HANSEN_SCALE_M,
            'crs': HANSEN_CRS,
            'format': 'NPY'
        })
    
        response = requests.get(url)
        response.raise_for_status() 
    except ee.ee_exception.EEException as e:
        if "must be less than or equal to" in str(e):
            print('polygon too large!')
            return {"data": None, "mask": None, "coords": None, "transform": None, "error": "polygon too large"}
        return {"data": None, "mask": None, "coords": None, "transform": None, "error": str(e)}
        
    # 4. Instant Memory Mapping
    raw_array = np.load(io.BytesIO(response.content))
    height, width = raw_array.shape

    # Derive pixel size from snapped corners and actual array dims (handles EE off-by-one)
    pixel_width = (east - west) / width
    pixel_height = (north - south) / height
    true_transform = Affine.translation(west, north) * Affine.scale(
        pixel_width, -pixel_height
    )

    # 6. Tabular Conversion & Masking
    hansen_year = config.TEST_YEAR - 2000
    landsat_hansen_pd = pd.DataFrame(raw_array.flatten())
    
    treecover_mask = (landsat_hansen_pd['treecover2000'] > 50)
    lossyear_0_mask = (landsat_hansen_pd['lossyear'] == 0)
    lossyear_after_mask = (landsat_hansen_pd['lossyear'] > hansen_year)
    
    valid_mask = treecover_mask & (lossyear_0_mask | lossyear_after_mask)
    print(landsat_hansen_pd['lossyear'].sort_values().unique())

    data = landsat_hansen_pd[config.BANDS_IN_ORDER].copy()
    loss = (landsat_hansen_pd['lossyear'] == hansen_year)
    data['loss'] = loss
    print(data['loss'].value_counts())
    print(data.head())
    return {
        "data": data,
        "mask": valid_mask,
        "coords": (height, width),
        "transform": true_transform,
        "error": None,
    }


def mask_clouds_landsat(image:ee.Image):
    """Given an Image from the Landsat 8 Collection 2 Tier 1 dataset, applies a mask that removes 
    clouds and shadows."""

    # Select the Quality Assessment band
    qa = image.select('QA_PIXEL')

    # Bits 3 and 4 are Cloud and Cloud Shadow, respectively.
    # We create a mask where these bits are set to 0 (Clear)
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4

    # Combined mask: pixel must have neither cloud nor shadow
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
           qa.bitwiseAnd(shadow_bit).eq(0))

    # Apply the mask to the image (making clouds/shadows transparent)
    return image.updateMask(mask)


def add_coords(feature:ee.Feature):
    """Given a Feature, adds the longitude and latitude as properties."""
    return feature.set({
        'longitude': feature.geometry().coordinates().get(0),
        'latitude': feature.geometry().coordinates().get(1)
    })


def get_bands(year: int, month: int, day: int) -> ee.ImageCollection:
    """Gets the Landsat 8 Collection 2 Tier 1 image bands from (year, month, day) to (year+1, month, day) for the provided geometry."""
    return ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterDate(ee.Date.fromYMD(year, month, day), ee.Date.fromYMD(year+1, month, day)) \
        .map(mask_clouds_landsat)

def get_indices(img_collection: ee.ImageCollection) -> ee.Image:
    """Given an ImageCollection from the Landsat 8 Collection 2 Tier 1 dataset, creates an image with the median
    Red, Near Infrared, Shortwave Infrared 1 and 2 spectral bands, as well as NDVI, NBR, NDMI indices.
    Learn more about spectral indices here: https://www.geo.university/pages/spectral-indices-in-remote-sensing-and-how-to-interpret-them"""
    # Select original bands
    selected_bands = img_collection.select(['SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).median()
    # Calculate indices using normalizedDifference
    ndvi = selected_bands.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI') # NIR - Red (SR_B5 - SR_B4)
    nbr = selected_bands.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR')   # NIR - SWIR2 (SR_B5 - SR_B7)
    ndmi = selected_bands.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI') # NIR - SWIR1 (SR_B5 - SR_B6)

    # Return original bands plus calculated indices
    return selected_bands.addBands([ndvi, nbr, ndmi])


def process_yearly_landsat(year:int, month:int, day:int, base_year:int, base_month:int, base_day:int)->ee.Image:
    """Given year, month, day and base_year, base_month, base_day, gets bands, indices, and year-to-base_year deltas of the Landsat 8 Collection 2 Tier 1 dataset as per get_indices(), from year to base_year for the provided location (a Geometry)"""
    # gets lagged spectral bands, spectral indices, and deltas from year to base_year
    start_date = ee.Date.fromYMD(year, month, day)


    # Process the target year
    img = get_indices(get_bands(year, month, day))
    last_year_img = get_indices(get_bands(base_year, base_month, base_day))
    if year != base_year:
        deltas = img.subtract(last_year_img).rename(img.bandNames().map(lambda n: ee.String(n).cat('_delta')))
        img = img.addBands(deltas)


    # Calculate Lags for labeling - Added .toInt() to avoid '.0' in band names
    lag = ee.Number(base_year).subtract(year)
    lag_suffix = ee.String('_lag').cat(lag.toInt().format())

    # Rename year bands
    img_renamed = img.rename(img.bandNames().map(lambda n: ee.String(n).cat(lag_suffix)))

    return img_renamed \
                .set('year', year) \
                .set('lag', lag) \
                .set('base_year', base_year) \
                .set('system:time_start', start_date.millis())