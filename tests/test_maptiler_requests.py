import asyncio
import io

import httpx
from PIL import Image

import app.services.imagery as imagery
from app.services.imagery import download_maptiler_area_tiles


def _jpeg_bytes(size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color=(120, 200, 150)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_maptiler_trims_api_key_and_uses_referer(tmp_path, monkeypatch):
    sample_image = _jpeg_bytes(size=512)
    requests_made: list[dict] = []

    class DummyResponse:
        def __init__(self, url: str, params: dict | None, headers: dict | None):
            self._url = httpx.URL(url, params=params)
            self.content = sample_image
            self.headers = {"Content-Type": "image/jpeg"}
            self.status_code = 200

        def raise_for_status(self) -> None:  # pragma: no cover - included for interface parity
            return None

        @property
        def url(self) -> httpx.URL:
            return self._url

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            self._calls = requests_made

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, params=None, headers=None):
            call = {"url": url, "params": params or {}, "headers": headers or {}}
            self._calls.append(call)
            return DummyResponse(url, params, headers)

    monkeypatch.setenv("MAPTILER_API_KEY", " test-key \n")
    monkeypatch.setenv("MAPTILER_REFERER", "https://example.test/app")
    monkeypatch.setenv("IMAGERY_REQUEST_DELAY", "0")
    monkeypatch.setenv("IMAGERY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(imagery, "_tile_cache", None)
    monkeypatch.setattr(imagery, "_last_remote_request_at", None)
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    tiles, failures = asyncio.run(
        download_maptiler_area_tiles(
            north=0.03,
            south=0.0,
            east=0.03,
            west=0.0,
            dim=0.03,
            output_dir=tmp_path,
        )
    )

    assert failures == []
    assert len(tiles) == 1
    assert requests_made, "Expected at least one MapTiler request"

    assert all(call["params"].get("key") == "test-key" for call in requests_made)
    assert all(
        call["headers"].get("Referer") == "https://example.test/app" for call in requests_made
    )
    assert all("tiles/satellite" in call["url"] for call in requests_made)

    tile = tiles[0]
    assert tile.pixel_size >= 512
    assert tile.source_url

    with Image.open(tile.path) as image:
        assert image.size == (tile.pixel_size, tile.pixel_size)
        assert image.getpixel((0, 0)) != image.getpixel((image.width - 1, image.height - 1))


def test_maptiler_reuses_cached_tiles(tmp_path, monkeypatch):
    sample_image = _jpeg_bytes(size=256)
    call_counter = {"count": 0}

    class DummyResponse:
        def __init__(self, url: str, params: dict | None, headers: dict | None):
            self._url = httpx.URL(url, params=params)
            self.content = sample_image
            self.headers = {"Content-Type": "image/jpeg"}
            self.status_code = 200

        def raise_for_status(self) -> None:  # pragma: no cover - interface parity
            return None

        @property
        def url(self) -> httpx.URL:
            return self._url

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, params=None, headers=None):
            call_counter["count"] += 1
            return DummyResponse(url, params, headers)

    monkeypatch.setenv("MAPTILER_API_KEY", "cache-test")
    monkeypatch.delenv("MAPTILER_REFERER", raising=False)
    monkeypatch.setenv("IMAGERY_REQUEST_DELAY", "0")
    monkeypatch.setenv("IMAGERY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(imagery, "_tile_cache", None)
    monkeypatch.setattr(imagery, "_last_remote_request_at", None)
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    tiles_first, failures_first = asyncio.run(
        download_maptiler_area_tiles(
            north=0.03,
            south=0.0,
            east=0.03,
            west=0.0,
            dim=0.03,
            output_dir=first_dir,
        )
    )

    assert failures_first == []
    assert len(tiles_first) == 1
    assert call_counter["count"] > 0

    initial_calls = call_counter["count"]

    tiles_second, failures_second = asyncio.run(
        download_maptiler_area_tiles(
            north=0.03,
            south=0.0,
            east=0.03,
            west=0.0,
            dim=0.03,
            output_dir=second_dir,
        )
    )

    assert failures_second == []
    assert len(tiles_second) == 1
    assert call_counter["count"] == initial_calls
