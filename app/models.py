from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Field, SQLModel


class AnalysisResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    image_filename: str
    prompt: str
    caption: str
    unusual_summary: str
    detection_payload: str = Field(default="[]", description="JSON encoded detections")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def detections(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.detection_payload)
        except json.JSONDecodeError:
            return []

    def detection_labels(self) -> str:
        detected = [item.get("object") for item in self.detections() if item.get("object")]
        return ", ".join(detected)
