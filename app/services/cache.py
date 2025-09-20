from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict


class TileCache:
    """Simple on-disk cache for storing downloaded imagery tiles."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, key: str, extension: str, destination: Path) -> Dict[str, Any] | None:
        """Copy a cached tile to ``destination`` and return its metadata."""

        cache_path = self._path(key, extension, ensure_parent=False)
        if not cache_path.exists():
            return None

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_path, destination)
        return self._read_metadata(cache_path)

    def store(self, key: str, extension: str, content: bytes, metadata: Dict[str, Any]) -> Path:
        """Persist ``content`` and ``metadata`` under the provided cache ``key``."""

        cache_path = self._path(key, extension, ensure_parent=True)
        cache_path.write_bytes(content)
        self._write_metadata(cache_path, metadata)
        return cache_path

    def _path(self, key: str, extension: str, *, ensure_parent: bool) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        directory = self.root / digest[:2] / digest[2:4]
        if ensure_parent:
            directory.mkdir(parents=True, exist_ok=True)
        extension = self._normalize_extension(extension)
        return directory / f"{digest}{extension}"

    def _metadata_path(self, cache_path: Path) -> Path:
        return cache_path.parent / f"{cache_path.name}.json"

    def _read_metadata(self, cache_path: Path) -> Dict[str, Any]:
        metadata_path = self._metadata_path(cache_path)
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _write_metadata(self, cache_path: Path, metadata: Dict[str, Any]) -> None:
        metadata_path = self._metadata_path(cache_path)
        metadata_path.write_text(json.dumps(metadata, sort_keys=True))

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        return extension if extension.startswith(".") else f".{extension}"
