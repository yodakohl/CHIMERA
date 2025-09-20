from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Describe all unusual objects in this image."

OBJECT_DETECTION_THRESHOLD = 0.25
MAX_DETECTIONS = 20


class SatelliteAnalyzer:
    """Wrapper around vision-language models for satellite object discovery."""

    def __init__(self) -> None:
        logger.info("Loading captioning and VQA pipelines for satellite analysis")
        self.captioning = pipeline(
            "image-to-text", model="Salesforce/blip2-flan-t5-xl"
        )
        self.vqa = pipeline(
            "visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa"
        )
        self.object_detector = pipeline(
            "object-detection", model="facebook/detr-resnet-50"
        )

    def analyze(self, image_path: Path, prompt: str | None = None) -> Dict[str, object]:
        prompt = prompt or DEFAULT_PROMPT
        image = Image.open(image_path).convert("RGB")

        caption = self._generate_caption(image)
        unusual_summary = self._ask_question(prompt, image)
        detections = self._detect_objects(image)

        return {
            "prompt": prompt,
            "caption": caption,
            "unusual_summary": unusual_summary,
            "detections": detections,
        }

    def _generate_caption(self, image: Image.Image) -> str:
        result = self.captioning(image)
        if result and isinstance(result, list):
            return result[0].get("generated_text", "").strip()
        return ""

    def _ask_question(self, question: str, image: Image.Image) -> str:
        result = self.vqa(question=question, image=image)
        if isinstance(result, list) and result:
            answer = result[0].get("answer", "")
            return answer.strip()
        if isinstance(result, dict):
            return str(result.get("answer", "")).strip()
        return ""

    def _detect_objects(self, image: Image.Image) -> List[Dict[str, object]]:
        raw_detections = self.object_detector(image)
        detections: List[Dict[str, object]] = []
        for entry in raw_detections:
            score = float(entry.get("score") or 0.0)
            if score < OBJECT_DETECTION_THRESHOLD:
                continue
            label = str(entry.get("label", "")).strip()
            if not label:
                continue
            detection: Dict[str, object] = {
                "object": label,
                "confidence": round(score, 3),
            }
            box = entry.get("box")
            if isinstance(box, dict):
                detection["box"] = {
                    "xmin": int(box.get("xmin", 0)),
                    "ymin": int(box.get("ymin", 0)),
                    "xmax": int(box.get("xmax", 0)),
                    "ymax": int(box.get("ymax", 0)),
                }
            detections.append(detection)

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        if len(detections) > MAX_DETECTIONS:
            detections = detections[:MAX_DETECTIONS]
        return detections


@lru_cache()
def get_analyzer() -> SatelliteAnalyzer:
    return SatelliteAnalyzer()


def serialize_detections(detections: Sequence[Dict[str, object]]) -> str:
    return json.dumps(list(detections))
