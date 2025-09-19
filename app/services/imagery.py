from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import httpx

DEMO_API_KEY = "DEMO_KEY"
DEMO_KEY_TILE_LIMIT = 2

logger = logging.getLogger(__name__)

NASA_EARTH_IMAGERY_URL = "https://api.nasa.gov/planetary/earth/imagery"
MIN_DIM = 0.01
MAX_DIM = 0.5
MAX_TILES_PER_RUN = 50
REQUEST_TIMEOUT = httpx.Timeout(60.0)


@dataclass
class AreaTile:
    """Metadata for a satellite tile downloaded from the NASA imagery API."""

    lat: float
    lon: float
    path: Path
    source_url: str


async def download_nasa_area_tiles(
    *,
    north: float,
    south: float,
    east: float,
    west: float,
    dim: float,
    output_dir: Path,
    api_key: str,
    date: str | None = None,
) -> Tuple[List[AreaTile], List[str]]:
    """Download a grid of imagery tiles that cover the requested bounding box.

    Returns a tuple consisting of successfully downloaded tiles and a list of
    human-readable error messages for tiles that could not be fetched.
    """

    _validate_bounds(north=north, south=south, east=east, west=west)
    _validate_dim(dim)

    output_dir.mkdir(parents=True, exist_ok=True)

    lat_centers = _build_axis_centers(
        minimum=south,
        maximum=north,
        dim=dim,
        clamp_min=-90.0 + dim / 2,
        clamp_max=90.0 - dim / 2,
    )
    lon_centers = _build_axis_centers(
        minimum=west,
        maximum=east,
        dim=dim,
        clamp_min=-180.0 + dim / 2,
        clamp_max=180.0 - dim / 2,
    )

    total_tiles = len(lat_centers) * len(lon_centers)

    max_tiles_for_key = _max_tiles_for_api_key(api_key)
    if total_tiles > max_tiles_for_key:
        if _is_demo_key(api_key):
            raise ValueError(
                "The NASA DEMO_KEY may only be used for two imagery requests at a time. "
                f"The requested area would require {total_tiles} tiles. Reduce the coverage or "
                "use your own NASA API key."
            )
        raise ValueError(
            "Requested area requires "
            f"{total_tiles} tiles. Reduce coverage or increase the tile size to stay below "
            f"the limit of {MAX_TILES_PER_RUN} requests per scan."
        )

    tiles: List[AreaTile] = []
    failures: List[str] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for lat in lat_centers:
            for lon in lon_centers:
                params = {
                    "lat": lat,
                    "lon": lon,
                    "dim": dim,
                    "api_key": api_key,
                }
                if date:
                    params["date"] = date

                try:
                    response = await client.get(NASA_EARTH_IMAGERY_URL, params=params)
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = _short_error_detail(exc.response.text)
                    failures.append(
                        f"lat {lat:.4f}, lon {lon:.4f}: {exc.response.status_code} {detail}"
                    )
                    logger.warning(
                        "NASA imagery request failed with status %s: %s",
                        exc.response.status_code,
                        detail,
                    )
                    continue
                except httpx.RequestError as exc:
                    failures.append(f"lat {lat:.4f}, lon {lon:.4f}: {exc}")
                    logger.warning("NASA imagery request error: %s", exc)
                    continue

                filename = _tile_filename(lat, lon, date)
                tile_path = output_dir / filename
                tile_path.write_bytes(response.content)
                tiles.append(
                    AreaTile(
                        lat=lat,
                        lon=lon,
                        path=tile_path,
                        source_url=str(response.url),
                    )
                )

    return tiles, failures


def _validate_bounds(*, north: float, south: float, east: float, west: float) -> None:
    if north <= south:
        raise ValueError("North latitude must be greater than south latitude.")
    if east <= west:
        raise ValueError("East longitude must be greater than west longitude.")
    if north > 90.0 or south < -90.0:
        raise ValueError("Latitudes must be within -90 and 90 degrees.")
    if east > 180.0 or west < -180.0:
        raise ValueError("Longitudes must be within -180 and 180 degrees.")
    if north - south > 180.0:
        raise ValueError("Latitude span is too large for a single scan. Please reduce coverage.")
    if east - west > 360.0:
        raise ValueError("Longitude span cannot exceed 360 degrees.")


def _validate_dim(dim: float) -> None:
    if not (MIN_DIM <= dim <= MAX_DIM):
        raise ValueError(
            f"Tile size (dim) must be between {MIN_DIM} and {MAX_DIM} degrees as required by the NASA API."
        )


def _build_axis_centers(
    *, minimum: float, maximum: float, dim: float, clamp_min: float, clamp_max: float
) -> List[float]:
    if maximum - minimum <= 0:
        center = _clamp((maximum + minimum) / 2, clamp_min, clamp_max)
        return [center]

    tile_count = max(1, math.ceil((maximum - minimum) / dim))
    centers: List[float] = []
    for index in range(tile_count):
        center = minimum + dim * (index + 0.5)
        if center > maximum:
            center = maximum - dim / 2
        center = _clamp(center, clamp_min, clamp_max)
        if centers and abs(center - centers[-1]) < 1e-6:
            continue
        centers.append(center)

    if not centers:
        centers.append(_clamp((maximum + minimum) / 2, clamp_min, clamp_max))

    return centers


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(min(value, maximum), minimum)


def _tile_filename(lat: float, lon: float, date: str | None) -> str:
    lat_token = _sanitize_coord(lat)
    lon_token = _sanitize_coord(lon)
    if date:
        date_token = date.replace("-", "")
        return f"nasa_{date_token}_{lat_token}_{lon_token}.png"
    return f"nasa_{lat_token}_{lon_token}.png"


def _sanitize_coord(value: float) -> str:
    token = f"{value:+.4f}".replace("+", "p").replace("-", "m").replace(".", "_")
    return token


def _short_error_detail(detail: str) -> str:
    detail = detail.strip()
    if len(detail) > 160:
        return f"{detail[:157]}..."
    return detail or "(no detail)"


def _max_tiles_for_api_key(api_key: str) -> int:
    if _is_demo_key(api_key):
        return min(MAX_TILES_PER_RUN, DEMO_KEY_TILE_LIMIT)
    return MAX_TILES_PER_RUN


def _is_demo_key(api_key: str) -> bool:
    return (api_key or "").strip().upper() == DEMO_API_KEY
