from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class GeoJSONFeature(BaseModel):
    type: str
    geometry: Dict[str, Any]
    properties: Optional[Dict[str, Any]] = None

class PredictRequest(BaseModel):
    """GeoJSON feature plus the prediction year (replaces global config.TEST_YEAR)."""
    year: int = Field(..., ge=2001, le=2026)
    feature: GeoJSONFeature