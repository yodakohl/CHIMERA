from __future__ import annotations

import torch
from PIL import Image

from app.services.analyzer import SatelliteAnalyzer


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.images: list[Image.Image] = []

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        content = conversation[0]["content"]
        assert content[0]["type"] == "image"
        return f"<image>\nUser: {content[1]['text']}\nAssistant:"

    def __call__(self, *, text, images, padding, return_tensors):
        self.texts = list(text)
        self.images = list(images)
        assert padding is True
        assert return_tensors == "pt"
        return _FakeInputs({"input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]])})

    def batch_decode(self, generated_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return ["Assistant: first tile summary", "second tile summary"]


class _FakeModel:
    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 128
        assert kwargs["do_sample"] is False
        return torch.tensor([[1, 2, 3, 10, 11], [1, 2, 0, 12, 13]])


def test_resolve_backend_accepts_smolvlm_aliases():
    assert SatelliteAnalyzer._resolve_backend("smolvlm") == "smolvlm"
    assert SatelliteAnalyzer._resolve_backend(" SmolVLM2 ") == "smolvlm2"


def test_smolvlm_batch_generation_uses_processor_images():
    analyzer = SatelliteAnalyzer.__new__(SatelliteAnalyzer)
    analyzer.smolvlm_processor = _FakeProcessor()
    analyzer.smolvlm_model = _FakeModel()
    analyzer.smolvlm_device = torch.device("cpu")

    images = [
        Image.new("RGB", (8, 8), color=(10, 20, 30)),
        Image.new("RGB", (8, 8), color=(40, 50, 60)),
    ]
    result = analyzer._generate_vlm_text_batch("Describe the scene.", images)

    assert result == ["first tile summary", "second tile summary"]
    assert analyzer.smolvlm_processor.images == images
    assert all("Describe the scene." in text for text in analyzer.smolvlm_processor.texts)
