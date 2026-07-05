import shapely
import ee
import numpy as np
import pandas as pd 
from shapely.geometry import box
import os
import requests
import io
from dotenv import load_dotenv

TROPIC_LAT = 23.5
BOREAL_LAT = 50.0 
MAX_PIXELS = 262144
BANDS_IN_ORDER = ["NBR_delta_lag1","NBR_lag1","NDMI_delta_lag1","NDMI_lag1","NDVI_delta_lag1","NDVI_lag1","SR_B4_delta_lag1","SR_B4_lag1","SR_B5_delta_lag1","SR_B5_lag1","SR_B6_delta_lag1","SR_B6_lag1","SR_B7_delta_lag1","SR_B7_lag1","NBR_lag0","NDMI_lag0","NDVI_lag0","SR_B4_lag0","SR_B5_lag0","SR_B6_lag0","SR_B7_lag0"] # in the same order as xgboost's features


load_dotenv()
project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID") # insert your id here

ee.Authenticate(force=False)
ee.Initialize(project=project_id)

def get_data_bbox(polygon: shapely.Polygon, TEST_YEAR: int = 24)->np.ndarray:
    """
    Creates a GeoJSON of the bounding box containing the polygon, extracts the landsat image bands, filters by Hansen treecover and lossyear data,
    and returns a pandas DataFrame with the landsat image data, and nan where treecover < 50 or if it's been deforested before TEST_YEAR
    
    Args:
        polygon: A Shapely Polygon geometry
        TEST_YEAR: Year threshold for tree loss (e.g., 24 for year 2024).
                           Pixels with lossyear <= this value are marked as -1.
        
    Returns:
        tuple of the form (data, mask, (height, width)) where:
        "data": A pandas DataFrame with the landsat image bands
        "mask": A numpy array with the same length as data, True only if treecover > 50 and lossyear = 0 or lossyear > TEST_YEAR
        "(height, width)": a tuple of the initial bounding box, for reconverting each row in the dataframe to pixel in the polygon
    """
    # Step 1: Create bounding box from polygon
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    bbox = box(min_lon, min_lat, max_lon, max_lat)
    
    # Step 2: Convert bbox to GeoJSON format
    bbox_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ]]
    }

    # Step 3: Create Earth Engine geometry from bbox GeoJSON
    ee_geometry = ee.Geometry(bbox_geojson)


    # extract landsat bands
    landsat_image_lag = process_yearly_landsat(TEST_YEAR-1, 1, 1, TEST_YEAR, 1, 1)
    landsat_image_current = process_yearly_landsat(TEST_YEAR, 1, 1, TEST_YEAR, 1, 1)
    landsat_image = landsat_image_current.addBands(landsat_image_lag)

    # extract hansen bands for filtering
    hansen = ee.Image('UMD/hansen/global_forest_change_2025_v1_13')

    # reshape the landsat image to hansen level
    target_projection = hansen.select('treecover2000').projection()
    print('landsat projection:', landsat_image.projection().getInfo())
    print('target projection:', target_projection.getInfo())
    print('before resampling')
    landsat_image = landsat_image.resample('bilinear').reproject(crs=target_projection)
    print('after resampling, before rect sampling')


    combined_image = landsat_image.addBands([
        hansen.select('treecover2000'), 
        hansen.select('lossyear')
    ])
    native_scale = target_projection.nominalScale().getInfo()
    crs_val = target_projection.crs().getInfo()
    
    # 4. Request the raw binary NPY file URL
    # Because we already aligned Landsat, this download will be mathematically pristine.
    url = combined_image.getDownloadURL({
        'region': ee_geometry,
        'scale': native_scale,
        'crs': crs_val,
        'format': 'NPY'
    })
    
    # 5. Execute the binary download (bypasses JSON text bloat)
    response = requests.get(url)
    response.raise_for_status() 
    
    # 6. Load directly into C-level contiguous memory
    raw_array = np.load(io.BytesIO(response.content))
    height, width = raw_array.shape
    test_landsat_pd = pd.DataFrame(raw_array.flatten())
    # print(test_landsat_pd)
    treecover_mask = (test_landsat_pd['treecover2000'] > 50)
    lossyear_0_mask = (test_landsat_pd['lossyear'] == 0)
    lossyear_after_mask = (test_landsat_pd['lossyear'] > TEST_YEAR-2000)
    # print(treecover_mask)
    # valid_mask = (test_landsat_pd['treecover2000'] > 50) & (test_landsat_pd['lossyear'] == 0 | test_landsat_pd['lossyear'] > TEST_YEAR-2000)
    return test_landsat_pd[BANDS_IN_ORDER], treecover_mask & (lossyear_0_mask | lossyear_after_mask), (height, width) 



    # # sample ee_gemetry from landsat_image
    # landsat_feature = landsat_image.sampleRectangle(region=ee_geometry, defaultValue=0)
    # try:
    #     # This is where the network call actually happens.
    #     # If it's too big, Earth Engine will instantly reject it here.
    #     feature_dict = landsat_feature.getInfo()['properties']
        
    # except ee.ee_exception.EEException as e:
    #     # Catch the exact limit error natively
    #     if "Too many pixels" in str(e):
    #         print("Payload too large. Rejecting.")
    #         return None, None, None
    #     else:
    #         # If it's a different Earth Engine error, raise it so you know
    #         raise e
    band_names_in_order = ["NBR_delta_lag1","NBR_lag1","NDMI_delta_lag1","NDMI_lag1","NDVI_delta_lag1","NDVI_lag1","SR_B4_delta_lag1","SR_B4_lag1","SR_B5_delta_lag1","SR_B5_lag1","SR_B6_delta_lag1","SR_B6_lag1","SR_B7_delta_lag1","SR_B7_lag1","NBR_lag0","NDMI_lag0","NDVI_lag0","SR_B4_lag0","SR_B5_lag0","SR_B6_lag0","SR_B7_lag0"] # in the same order as xgboost's features
    # band_arrays = [np.array(feature_dict[b], dtype=np.float32) for b in band_names]
    # height, width = band_arrays[0].shape # essential to save this because we're about to transform the 2d rectangle into a 1d list for the model
    # print('height:', height, 'width:', width)
    # landsat_pd = pd.DataFrame(np.column_stack([arr.ravel() for arr in band_arrays]))
    # landsat_pd.columns = band_names
    
    # Step 4: Load Hansen Global Forest Change treecover dataset
    # Hansen dataset has annual treecover loss and gain data
    # Using the treecover percentage layer and lossyear layer
    treecover = hansen.select('treecover2000')  # Baseline treecover in year 2000
    lossyear = hansen.select('lossyear')  # Year of tree loss
    
    # Step 5: Sample the treecover and lossyear data at 30m resolution (Landsat resolution)
    treecover_clipped = treecover.clipToCollection(ee.FeatureCollection([landsat_feature]))
    lossyear_clipped = lossyear.clipToCollection(ee.FeatureCollection([landsat_feature]))
    
    # Step 6: Convert to numpy arrays
    treecover_array = treecover_clipped.sampleRectangle(
        defaultValue=0,
        region=landsat_feature.geometry(),
    )
    
    lossyear_array = lossyear_clipped.sampleRectangle(
        defaultValue=0,
        region=landsat_feature.geometry(),
    )
    
    # Step 7: Extract the bands
    treecover_data = treecover_array.get('treecover2000').getInfo()
    # print(treecover_array)
    treecover_np = np.array(treecover_data, dtype=np.float32)
    # print(treecover_np)
    lossyear_data = lossyear_array.get('lossyear').getInfo()
    lossyear_np = np.array(lossyear_data, dtype=np.int32)
    
    # Step 8: Set pixels to -1 if:
    # - treecover < 50, OR
    # - lossyear > 0 AND lossyear <= loss_year_threshold (already deforested)
    treecover_mask = (treecover_np > 50).ravel()
    lossyear_mask = ((lossyear_np == 0) | (lossyear_np > TEST_YEAR-2000)).ravel()
    valid_mask = treecover_mask & lossyear_mask
    # landsat_pd = landsat_pd.where(pd.Series(treecover_mask == False, index=landsat_pd.index), axis=0).where(pd.Series(lossyear_mask == False, index=landsat_pd.index), axis=0)

    
    return landsat_pd, valid_mask, (height, width)




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
    # print('Available bands:', img_collection.first().bandNames().getInfo())
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