import numpy as np
from rasterio.features import geometry_mask

from config import TROPIC_LAT, BOREAL_LAT, THRESHOLD

def route_model_and_predict(centroid, data, models):


    if -TROPIC_LAT < centroid.y < TROPIC_LAT:
        # tropical
        active_model=models['tropical']
    elif TROPIC_LAT < centroid.y < BOREAL_LAT:
        # temperate north
        active_model=models['temperate']
    elif -90 < centroid.y < -TROPIC_LAT:
        # temperate south
        active_model=models['temperate']
    elif BOREAL_LAT < centroid.y < 90:
        # boreal
        active_model=models['boreal']

    raw_probs = active_model.predict_proba(data)[:, 1]
    predictions = np.where(raw_probs > 0.5, 1, -1)
    return predictions


def values_to_raster(values, mask, height, width, polygon, transform):
    """Scatter 1D masked values onto a 2D grid, zeroing pixels outside the polygon.

    Args:
        values: 1D array aligned with True entries in ``mask`` (e.g. predictions or Hansen loss).
        mask: Boolean mask over the full flattened grid (length height * width).
        height, width: Output raster dimensions.
        polygon: Shapely polygon used to clip the raster to the user-drawn region.
        transform: Affine transform matching the raster grid.

    Returns:
        2D float32 array of shape (height, width); 0 outside the mask/polygon.
    """
    total_pixels = height * width
    flat_output = np.full(total_pixels, 0, dtype=np.float32)
    flat_output[mask] = values
    final_2d_map = flat_output.reshape((height, width))

    polygon_mask = geometry_mask(
        [polygon],
        out_shape=(height, width),
        transform=transform,
        invert=True,  # pixels INSIDE the polygon are True
    )
    final_2d_map[~polygon_mask] = 0
    return final_2d_map
