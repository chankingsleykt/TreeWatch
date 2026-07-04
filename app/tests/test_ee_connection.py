import pytest
from app.ee_connection import filter_from_bbox
import shapely
import numpy as np

def test_filters_lossyear():
   loss_24 = shapely.Polygon([   
    (-105.96113663500978,56.86698642911384),
    (-105.95753174609376,56.86698642911384),
    (-105.95753174609376,56.86858146188767),
    (-105.96113663500978,56.86858146188767),
    (-105.96113663500978,56.86698642911384)])
   # print(filter_from_bbox(loss_24, 24))
   assert np.any(filter_from_bbox(loss_24, 24)==-2)


def test_filters_treecover():
    lake = shapely.Polygon([
     (-105.8474435766621,56.83390507696947),
     (-105.84143542846874,56.83390507696947),
     (-105.84143542846874,56.836534476984184),
     (-105.8474435766621,56.836534476984184),
     (-105.8474435766621,56.83390507696947)])
    # print(filter_from_bbox(lake, 21))
    assert (filter_from_bbox(lake, 21)==-1).all()
