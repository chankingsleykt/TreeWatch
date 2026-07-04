import shapely
import ee
import numpy as np
from shapely.geometry import box
import os
from dotenv import load_dotenv

TROPIC_LAT = 23.5
BOREAL_LAT = 50.0 
MAX_PIXELS = 262144
TEST_YEAR=24

load_dotenv()
project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID") # insert your id here

ee.Authenticate(force=False)
ee.Initialize(project=project_id)

def filter_from_bbox(polygon: shapely.Polygon, loss_year_threshold: int = 24):
    """
    Creates a GeoJSON of the bounding box containing the polygon,
    queries Google Earth Engine for Hansen treecover and lossyear data,
    and returns a numpy array of -1 where:
    - treecover < 50%, OR
    - lossyear <= loss_year_threshold (already deforested)
    
    Args:
        polygon: A Shapely Polygon geometry
        loss_year_threshold: Year threshold for tree loss (e.g., 24 for year 2024).
                           Pixels with lossyear <= this value are marked as -1.
        
    Returns:
        treecover_array: A numpy array with -1 where conditions are met, 
                        and the treecover percentage (0-100) elsewhere
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

    pixelCount = ee.Image.constant(1).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=ee_geometry,
        scale=30,
        maxPixels=2*MAX_PIXELS,
        bestEffort=True
    )
    print(pixelCount.get('constant').getInfo())
    if pixelCount.get('constant').getInfo() >= MAX_PIXELS:
        print('too many:', pixelCount.get('constant').getInfo())
        return
    
    # Step 4: Load Hansen Global Forest Change treecover dataset
    # Hansen dataset has annual treecover loss and gain data
    # Using the treecover percentage layer and lossyear layer
    hansen = ee.Image('UMD/hansen/global_forest_change_2025_v1_13')
    treecover = hansen.select('treecover2000')  # Baseline treecover in year 2000
    lossyear = hansen.select('lossyear')  # Year of tree loss
    
    # Step 5: Sample the treecover and lossyear data at 30m resolution (Landsat resolution)
    treecover_clipped = treecover.clipToCollection(ee.FeatureCollection([ee_geometry]))
    lossyear_clipped = lossyear.clipToCollection(ee.FeatureCollection([ee_geometry]))
    
    # Step 6: Convert to numpy arrays
    treecover_array = treecover_clipped.sampleRectangle(
        defaultValue=0,
        region=ee_geometry,
    )
    
    lossyear_array = lossyear_clipped.sampleRectangle(
        defaultValue=0,
        region=ee_geometry,
    )
    
    # Step 7: Extract the bands
    treecover_data = treecover_array.get('treecover2000').getInfo()
    treecover_np = np.array(treecover_data, dtype=np.float32)
    
    lossyear_data = lossyear_array.get('lossyear').getInfo()
    lossyear_np = np.array(lossyear_data, dtype=np.int32)
    
    # Step 8: Set pixels to -1 if:
    # - treecover < 50, OR
    # - lossyear > 0 AND lossyear <= loss_year_threshold (already deforested)
    treecover_mask = (treecover_np < 50)
    lossyear_mask = ((lossyear_np > 0) & (lossyear_np <= loss_year_threshold))
    result_array = treecover_np.copy()
    result_array[treecover_mask] = -1
    result_array[lossyear_mask] = -2
    
    return result_array

def eliminate_non_forest_and_route_model(polygon:shapely.Polygon):
    centroid = polygon.centroid
    forest_pixels = filter_from_bbox(polygon, TEST_YEAR)
    print(forest_pixels)
    if -TROPIC_LAT < centroid.y < TROPIC_LAT:
        # use tropical model
        print('tropical')
    elif TROPIC_LAT < centroid.y < BOREAL_LAT:
        # use temperate model
        print('temperate north')
    elif -90 < centroid.y < -TROPIC_LAT:
        # use temperate model
        print('temperate south')
    elif BOREAL_LAT < centroid.y < 90:
        # use boreal model
        print('boreal')