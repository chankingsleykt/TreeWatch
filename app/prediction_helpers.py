from shapely.geometry import shape
from ee_connection import get_data_bbox
import numpy as np

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