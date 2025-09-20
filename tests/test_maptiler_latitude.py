import asyncio

import pytest

from app.services.imagery import MAPTILER_LATITUDE_LIMIT, download_maptiler_area_tiles


def test_maptiler_rejects_out_of_range_latitudes(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPTILER_API_KEY", "test-key")

    with pytest.raises(ValueError) as exc:
        asyncio.run(
            download_maptiler_area_tiles(
                north=MAPTILER_LATITUDE_LIMIT + 1.0,
                south=MAPTILER_LATITUDE_LIMIT - 0.5,
                east=10.0,
                west=9.0,
                dim=0.1,
                output_dir=tmp_path,
            )
        )

    message = str(exc.value)
    assert "latitude" in message.lower()
    assert "gibs" in message.lower()
