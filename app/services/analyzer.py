from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Describe all unusual objects in this image."

OBJECT_DETECTION_THRESHOLD = 0.25
MAX_DETECTIONS = 20

CAPTION_GENERATE_KWARGS = {"max_new_tokens": 80, "num_beams": 5}
SMOLVLM_GENERATE_KWARGS = {"max_new_tokens": 128, "do_sample": False}
SMOLVLM_CAPTION_PROMPT = (
    "Describe this satellite image for an infrastructure analyst. Mention visible "
    "land use, buildings, roads, vegetation, water, and any unusual objects."
)
DEFAULT_VLM_BACKEND = "legacy"
SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
LEGACY_BACKENDS = {"legacy", "blip", "blip2", "blip-vilt"}
SMOLVLM_BACKENDS = {"smolvlm", "smolvlm2"}

DETAILED_DESCRIPTION_QUESTIONS: List[Tuple[str, str, int]] = [
    (
        "overview",
        (
            "Provide a two-sentence analytic description summarizing the overall land "
            "use, major structures, vegetation, and transportation features visible in "
            "this satellite image."
        ),
        6,
    ),
    (
        "structures",
        (
            "Describe any buildings or man-made structures in this satellite image, "
            "mentioning their shape or placement. Respond with a complete descriptive "
            "sentence."
        ),
        4,
    ),
    (
        "fields",
        (
            "Describe the agricultural fields or land parcels in this satellite scene, "
            "noting their condition or crop state. Respond with a complete descriptive "
            "sentence."
        ),
        4,
    ),
    (
        "vegetation",
        (
            "Describe the vegetation, tree cover, or natural terrain visible in this "
            "satellite view. Respond with a complete descriptive sentence."
        ),
        3,
    ),
    (
        "roads",
        (
            "Describe any roads, paths, or access routes visible in this satellite image "
            "and how they connect the scene. Respond with a complete descriptive "
            "sentence."
        ),
        3,
    ),
    (
        "water",
        (
            "Describe any bodies of water, ponds, or drainage features visible in this "
            "satellite image. If none are visible, say 'No water features are visible.' "
            "Respond with a complete descriptive sentence."
        ),
        3,
    ),
]

GENERIC_DETAIL_ANSWERS = {
    "",
    "n/a",
    "na",
    "none",
    "nothing",
    "unknown",
    "not sure",
    "unsure",
    "unclear",
    "can't tell",
    "cannot tell",
    "can't see",
    "not visible",
    "no idea",
}


class SatelliteAnalyzer:
    """Wrapper around vision-language models for satellite object discovery."""

    def __init__(
        self,
        backend: str | None = None,
        smolvlm_model: str | None = None,
    ) -> None:
        self.backend = self._resolve_backend(
            backend if backend is not None else os.getenv("CHIMERA_VLM_BACKEND")
        )
        self.smolvlm_model_id = (smolvlm_model or "").strip() or None
        self.captioning = None
        self.vqa = None
        self.smolvlm_processor = None
        self.smolvlm_model = None
        self.smolvlm_device = None

        if self.backend in SMOLVLM_BACKENDS:
            self._load_smolvlm()
        else:
            logger.info("Loading BLIP-2 and VILT pipelines for satellite analysis")
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

        base_captions = self._generate_base_captions_batch(images)
        detail_answers = self._collect_detail_answers_batch(images)
        unusual_summaries = self._ask_question_batch(prompt, images)
        detections_batch = self._detect_objects_batch(images)

        results: List[Dict[str, object]] = []
        for index in range(len(images)):
            base_caption = base_captions[index] if index < len(base_captions) else ""
            per_image_details = {
                key: answers[index]
                for key, answers in detail_answers.items()
                if index < len(answers)
            }
            caption = self._compose_detailed_caption(base_caption, per_image_details)
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
        base_caption = self._generate_base_caption(image)
        detail_answers = self._collect_detail_answers(image)
        return self._compose_detailed_caption(base_caption, detail_answers)

    def _generate_base_caption(self, image: Image.Image) -> str:
        if self.backend in SMOLVLM_BACKENDS:
            return self._generate_vlm_text(SMOLVLM_CAPTION_PROMPT, image)

        try:
            result = self._run_captioning(image)
        except Exception as exc:  # pragma: no cover - model failure
            logger.debug("Caption generation failed, returning empty caption: %s", exc)
            return ""
        return self._extract_caption(result)

    def _generate_base_captions_batch(self, images: Sequence[Image.Image]) -> List[str]:
        if not images:
            return []

        if self.backend in SMOLVLM_BACKENDS:
            return self._generate_vlm_text_batch(SMOLVLM_CAPTION_PROMPT, images)

        try:
            result = self._run_captioning(images)
        except Exception as exc:  # pragma: no cover - model failure
            logger.debug(
                "Batch captioning failed, falling back to per-image mode: %s", exc
            )
            return [self._generate_base_caption(image) for image in images]

        captions = self._normalize_batch_output(result, len(images), self._extract_caption)
        if captions is None:
            return [self._generate_base_caption(image) for image in images]
        return captions

    def _collect_detail_answers(self, image: Image.Image) -> Dict[str, str]:
        batch_answers = self._collect_detail_answers_batch([image])
        return {
            key: (values[0] if values else "") for key, values in batch_answers.items()
        }

    def _collect_detail_answers_batch(
        self, images: Sequence[Image.Image]
    ) -> Dict[str, List[str]]:
        answers: Dict[str, List[str]] = {}
        if not images:
            return answers
        if self.backend in SMOLVLM_BACKENDS:
            return answers

        for key, question, _ in DETAILED_DESCRIPTION_QUESTIONS:
            responses = self._ask_question_batch(question, images)
            answers[key] = responses

        return answers

    def _compose_detailed_caption(
        self, base_caption: str, detail_answers: Dict[str, str] | None = None
    ) -> str:
        detail_answers = detail_answers or {}

        overview_raw = detail_answers.get("overview", "")
        overview_sentence = self._normalize_detail_answer(overview_raw, minimum_words=6)
        base_sentence = self._normalize_main_caption(base_caption)

        sentences: List[str] = []
        if overview_sentence:
            sentences.append(overview_sentence)
            if base_sentence:
                base_core = base_sentence.rstrip(".?!").lower()
                if base_core and base_core not in overview_sentence.lower():
                    sentences.append(base_sentence)
        elif base_sentence:
            sentences.append(base_sentence)

        detail_sentences: List[str] = []
        seen: set[str] = set()
        for key, _question, minimum_words in DETAILED_DESCRIPTION_QUESTIONS:
            if key == "overview":
                continue
            raw_answer = detail_answers.get(key, "")
            normalized = self._normalize_detail_answer(
                raw_answer, minimum_words=minimum_words
            )
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen or (base_sentence and lowered == base_sentence.lower()):
                continue
            seen.add(lowered)
            detail_sentences.append(normalized)

        if detail_sentences:
            first_detail = detail_sentences[0]
            if sentences:
                sentences.append(f"Notable details: {first_detail}")
            else:
                sentences.append(first_detail)
            for extra in detail_sentences[1:]:
                sentences.append(extra)

        return " ".join(sentence.strip() for sentence in sentences if sentence.strip())

    @staticmethod
    def _normalize_main_caption(caption: str) -> str:
        text = str(caption or "").strip()
        if not text:
            return ""

        stripped = text.rstrip(".?! ")
        if not stripped:
            return ""

        lower = stripped.lower()
        if lower.startswith("this ") or lower.startswith("these "):
            sentence = stripped
        else:
            sentence = f"This satellite image shows {stripped}"

        sentence = sentence.strip()
        if sentence and not sentence[0].isupper():
            sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence

    @staticmethod
    def _normalize_detail_answer(answer: str, minimum_words: int = 3) -> str:
        text = str(answer or "").strip()
        if not text:
            return ""

        lowered = text.lower().strip()
        if lowered in GENERIC_DETAIL_ANSWERS or lowered in {"yes", "no"}:
            return ""

        stripped = text.strip()
        stripped = stripped.rstrip(".?! ")
        if not stripped:
            return ""

        tokens = stripped.split()
        if len(tokens) < minimum_words and len(stripped) < 15:
            return ""

        normalized = stripped
        if normalized and not normalized[0].isupper():
            normalized = normalized[0].upper() + normalized[1:]
        if not normalized.endswith((".", "!", "?")):
            normalized += "."
        return normalized

    def _run_captioning(self, inputs):
        if self.captioning is None:
            raise RuntimeError("Captioning pipeline is not loaded for this backend")

        attempts = (
            {"generate_kwargs": CAPTION_GENERATE_KWARGS},
            CAPTION_GENERATE_KWARGS,
        )
        last_exception: Exception | None = None
        for kwargs in attempts:
            try:
                return self.captioning(inputs, **kwargs)
            except TypeError as exc:
                last_exception = exc
            except Exception as exc:  # pragma: no cover - model failure
                last_exception = exc
                logger.debug("Captioning attempt with kwargs %s failed: %s", kwargs, exc)
        if last_exception is not None:
            logger.debug("Falling back to default caption call: %s", last_exception)
        return self.captioning(inputs)

    def _ask_question(self, question: str, image: Image.Image) -> str:
        if self.backend in SMOLVLM_BACKENDS:
            return self._generate_vlm_text(question, image)
        if self.vqa is None:
            raise RuntimeError("VQA pipeline is not loaded for this backend")

        result = self.vqa(question=question, image=image)
        return self._extract_answer(result)

    def _ask_question_batch(
        self, question: str, images: Sequence[Image.Image]
    ) -> List[str]:
        if not images:
            return []

        if self.backend in SMOLVLM_BACKENDS:
            return self._generate_vlm_text_batch(question, images)
        if self.vqa is None:
            raise RuntimeError("VQA pipeline is not loaded for this backend")

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

    @staticmethod
    def _resolve_backend(raw_backend: str | None) -> str:
        backend = (raw_backend or DEFAULT_VLM_BACKEND).strip().lower()
        if not backend:
            return DEFAULT_VLM_BACKEND
        if backend in LEGACY_BACKENDS or backend in SMOLVLM_BACKENDS:
            return backend
        valid = sorted(LEGACY_BACKENDS | SMOLVLM_BACKENDS)
        raise ValueError(
            f"Unsupported CHIMERA_VLM_BACKEND '{raw_backend}'. Expected one of: {', '.join(valid)}"
        )

    def _load_smolvlm(self) -> None:
        model_id = (
            self.smolvlm_model_id or os.getenv("CHIMERA_SMOLVLM_MODEL", SMOLVLM_MODEL_ID)
        ).strip()
        if not model_id:
            model_id = SMOLVLM_MODEL_ID

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency mismatch
            raise RuntimeError(
                "SmolVLM requires transformers with AutoModelForImageTextToText support "
                "and torch installed. Run `pip install -e .` after updating dependencies."
            ) from exc

        logger.info("Loading SmolVLM backend %s for satellite analysis", model_id)
        self.smolvlm_processor = AutoProcessor.from_pretrained(model_id)
        self.smolvlm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_kwargs: Dict[str, object] = {}
        if self.smolvlm_device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16

        try:
            self.smolvlm_model = AutoModelForImageTextToText.from_pretrained(
                model_id, **model_kwargs
            )
        except TypeError:
            if "torch_dtype" in model_kwargs:
                model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            self.smolvlm_model = AutoModelForImageTextToText.from_pretrained(
                model_id, **model_kwargs
            )

        self.smolvlm_model.to(self.smolvlm_device)
        self.smolvlm_model.eval()

    def _generate_vlm_text(self, question: str, image: Image.Image) -> str:
        responses = self._generate_vlm_text_batch(question, [image])
        return responses[0] if responses else ""

    def _generate_vlm_text_batch(
        self, question: str, images: Sequence[Image.Image]
    ) -> List[str]:
        if not images:
            return []
        if (
            self.smolvlm_processor is None
            or self.smolvlm_model is None
            or self.smolvlm_device is None
        ):
            raise RuntimeError("SmolVLM backend is not loaded")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency mismatch
            raise RuntimeError("SmolVLM generation requires torch") from exc

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            for _image in images
        ]
        texts = [
            self.smolvlm_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conversation in conversations
        ]
        inputs = self.smolvlm_processor(
            text=texts,
            images=list(images),
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.smolvlm_device)
        with torch.no_grad():
            generated_ids = self.smolvlm_model.generate(
                **inputs,
                **SMOLVLM_GENERATE_KWARGS,
            )

        input_token_count = inputs["input_ids"].shape[-1]
        if generated_ids.shape[-1] > input_token_count:
            generated_ids = generated_ids[:, input_token_count:]

        decoded = self.smolvlm_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return [self._clean_generated_text(text) for text in decoded]

    @staticmethod
    def _clean_generated_text(text: str) -> str:
        cleaned = str(text or "").strip()
        for prefix in ("Assistant:", "assistant:", "ASSISTANT:", "Assistant", "assistant"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned


@lru_cache()
def get_analyzer(
    backend: str | None = None,
    smolvlm_model: str | None = None,
) -> SatelliteAnalyzer:
    normalized_backend = SatelliteAnalyzer._resolve_backend(
        backend if backend is not None else os.getenv("CHIMERA_VLM_BACKEND")
    )
    normalized_model = (smolvlm_model or "").strip() or None
    return SatelliteAnalyzer(normalized_backend, normalized_model)


def serialize_detections(detections: Sequence[Dict[str, object]]) -> str:
    return json.dumps(list(detections))
