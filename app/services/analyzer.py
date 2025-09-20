from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Sequence

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
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")

        caption = self._generate_caption(image)
        unusual_summary = self._ask_question(prompt, image)
        detections = self._detect_objects(image)

        return {
            "prompt": prompt,
            "caption": caption,
            "unusual_summary": unusual_summary,
            "detections": detections,
        }

    def analyze_many(
        self, image_paths: Sequence[Path], prompt: str | None = None
    ) -> List[Dict[str, object]]:
        prompt = prompt or DEFAULT_PROMPT
        if not image_paths:
            return []

        images: List[Image.Image] = []
        for path in image_paths:
            with Image.open(path) as raw_image:
                images.append(raw_image.convert("RGB"))

        captions = self._generate_captions_batch(images)
        unusual_summaries = self._ask_question_batch(prompt, images)
        detections_batch = self._detect_objects_batch(images)

        results: List[Dict[str, object]] = []
        for index in range(len(images)):
            caption = captions[index] if index < len(captions) else ""
            summary = (
                unusual_summaries[index] if index < len(unusual_summaries) else ""
            )
            detections = (
                detections_batch[index] if index < len(detections_batch) else []
            )
            results.append(
                {
                    "prompt": prompt,
                    "caption": caption,
                    "unusual_summary": summary,
                    "detections": detections,
                }
            )

        return results

    def _generate_caption(self, image: Image.Image) -> str:
        result = self.captioning(image)
        return self._extract_caption(result)

    def _generate_captions_batch(self, images: Sequence[Image.Image]) -> List[str]:
        if not images:
            return []

        try:
            result = self.captioning(images)
        except Exception as exc:  # pragma: no cover - model failure
            logger.debug("Batch captioning failed, falling back to per-image mode: %s", exc)
            return [self._generate_caption(image) for image in images]

        captions = self._normalize_batch_output(result, len(images), self._extract_caption)
        if captions is None:
            return [self._generate_caption(image) for image in images]
        return captions

    def _ask_question(self, question: str, image: Image.Image) -> str:
        result = self.vqa(question=question, image=image)
        return self._extract_answer(result)

    def _ask_question_batch(
        self, question: str, images: Sequence[Image.Image]
    ) -> List[str]:
        if not images:
            return []

        inputs = [{"question": question, "image": image} for image in images]

        try:
            result = self.vqa(inputs)
        except Exception as exc:  # pragma: no cover - model failure
            logger.debug("Batch VQA failed, falling back to per-image mode: %s", exc)
            return [self._ask_question(question, image) for image in images]

        answers = self._normalize_batch_output(result, len(images), self._extract_answer)
        if answers is None:
            return [self._ask_question(question, image) for image in images]
        return answers

    def _detect_objects(self, image: Image.Image) -> List[Dict[str, object]]:
        raw_detections = self.object_detector(image)
        return self._format_detections(raw_detections)

    def _detect_objects_batch(
        self, images: Sequence[Image.Image]
    ) -> List[List[Dict[str, object]]]:
        if not images:
            return []

        try:
            raw_batch = self.object_detector(images)
        except Exception as exc:  # pragma: no cover - model failure
            logger.debug(
                "Batch object detection failed, falling back to per-image mode: %s", exc
            )
            return [self._detect_objects(image) for image in images]

        detections = self._normalize_batch_output(
            raw_batch, len(images), self._format_detections
        )
        if detections is None:
            return [self._detect_objects(image) for image in images]
        return detections

    @staticmethod
    def _extract_caption(result) -> str:
        if isinstance(result, list):
            if not result:
                return ""
            first = result[0]
            return SatelliteAnalyzer._extract_caption(first)
        if isinstance(result, dict):
            text = result.get("generated_text") or result.get("caption") or ""
            return str(text).strip()
        return str(result or "").strip()

    @staticmethod
    def _extract_answer(result) -> str:
        if isinstance(result, list):
            if not result:
                return ""
            first = result[0]
            return SatelliteAnalyzer._extract_answer(first)
        if isinstance(result, dict):
            return str(result.get("answer", "")).strip()
        return str(result or "").strip()

    def _format_detections(self, raw_detections) -> List[Dict[str, object]]:
        if isinstance(raw_detections, tuple):
            raw_list = list(raw_detections)
        elif isinstance(raw_detections, list):
            raw_list = raw_detections
        elif isinstance(raw_detections, dict):
            raw_list = [raw_detections]
        else:
            return []

        detections: List[Dict[str, object]] = []
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
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

    def _normalize_batch_output(
        self,
        raw_batch,
        expected_len: int,
        extractor: Callable[[object], object],
    ) -> List[object] | None:
        if expected_len <= 0:
            return []

        if isinstance(raw_batch, tuple):
            raw_batch = list(raw_batch)

        if isinstance(raw_batch, list):
            if len(raw_batch) == expected_len:
                return [extractor(item) for item in raw_batch]
            if expected_len == 1:
                return [extractor(raw_batch)]
        elif expected_len == 1:
            return [extractor(raw_batch)]

        return None


@lru_cache()
def get_analyzer() -> SatelliteAnalyzer:
    return SatelliteAnalyzer()


def serialize_detections(detections: Sequence[Dict[str, object]]) -> str:
    return json.dumps(list(detections))
