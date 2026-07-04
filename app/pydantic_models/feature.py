from pydantic import BaseModel
from typing import Dict, Any

class GeoJSONFeature(BaseModel):
    type: str
    geometry: Dict[str, Any]
    properties: Dict[str, Any] = None