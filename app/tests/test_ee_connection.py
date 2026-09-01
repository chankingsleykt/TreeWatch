import pytest
from app.ee_connection import get_data_bbox
import shapely
import numpy as np
import pandas as pd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_bounds

def test_filters_lossyear():
   loss_24 = shapely.Polygon([   
    (-105.96113663500978,56.86698642911384),
    (-105.95753174609376,56.86698642911384),
    (-105.95753174609376,56.86858146188767),
    (-105.96113663500978,56.86858146188767),
    (-105.96113663500978,56.86698642911384)])
   result = get_data_bbox(loss_24)
   print(result["mask"])
   assert np.any(result["mask"] == False)


def test_filters_treecover():
    lake = shapely.Polygon([
     (-105.8474435766621,56.83390507696947),
     (-105.84143542846874,56.83390507696947),
     (-105.84143542846874,56.836534476984184),
     (-105.8474435766621,56.836534476984184),
     (-105.8474435766621,56.83390507696947)])
    result = get_data_bbox(lake)
    print(result["mask"])
    assert (result["mask"] == False).all()



def test_spatial_integrity():
    """
    Tests that the flattened Pandas DataFrame can be mathematically 
    mapped exactly back to the physical coordinates of the input Polygon.
    """
    # 1. Define a strict bounding box (e.g., a small patch in the Amazon)
    min_lon, min_lat = -65.0, -5.0
    max_lon, max_lat = -64.9, -4.9
    test_polygon = box(min_lon, min_lat, max_lon, max_lat)
    
    # 2. Run the extraction pipeline
    # (Requires Earth Engine to be initialized before running pytest)
    result = get_data_bbox(test_polygon)
    landsat_pd = result["data"]
    mask = result["mask"]
    height, width = result["coords"]
    
    # 3. Test DataFrame Integrity
    # The total number of rows must perfectly equal height * width
    total_pixels = height * width
    assert len(landsat_pd) == total_pixels, f"Data loss! Expected {total_pixels} rows, got {len(landsat_pd)}"
    
    # 4. Reconstruct the Spatial Anchor (Affine Transform)
    # This formula maps the grid dimensions back to the physical bounding box
    transform = from_bounds(
        west=min_lon, 
        south=min_lat, 
        east=max_lon, 
        north=max_lat, 
        width=width, 
        height=height
    )
    
    # 5. Test Spatial Pinning (Top-Left)
    # In a flattened C-contiguous array, Index 0 is the Top-Left pixel (Row 0, Col 0).
    # The affine transform of (0, 0) should equal exactly the North-West corner of the bounding box.
    top_left_lon, top_left_lat = transform * (0, 0)
    
    # We use pytest.approx to handle floating-point arithmetic micro-variances
    assert top_left_lon == pytest.approx(min_lon, rel=1e-5), "Top-Left Longitude drifted!"
    assert top_left_lat == pytest.approx(max_lat, rel=1e-5), "Top-Left Latitude drifted!"
    
    # 6. Test Spatial Pinning (Bottom-Right)
    # The bottom-right edge of the grid is at coordinate (width, height)
    # This should perfectly equal the South-East corner of the bounding box.
    bottom_right_lon, bottom_right_lat = transform * (width, height)
    
    assert bottom_right_lon == pytest.approx(max_lon, rel=1e-5), "Bottom-Right Longitude drifted!"
    assert bottom_right_lat == pytest.approx(min_lat, rel=1e-5), "Bottom-Right Latitude drifted!"

    print("\n[SUCCESS] Matrix can be perfectly folded back into the physical geographic bounds.")