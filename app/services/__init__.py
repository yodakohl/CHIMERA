"""Service utilities exposed by the ``app.services`` package."""

from .gpu_classifier import batch_classify_images, classify_images, classify_images_gpu

__all__ = ["classify_images_gpu", "classify_images", "batch_classify_images"]

