from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date as dt_date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import xml.etree.ElementTree as ET
from PIL import Image

logger = logging.getLogger(__name__)

GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GIBS_DEFAULT_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
GIBS_IMAGE_FORMAT = "image/png"
# The VIIRS "best" layer provides roughly 1 km per pixel resolution which equates to
# ~1024 pixels per degree at the equator. Requesting larger tiles simply produces
# upscaled imagery without exposing additional detail, so we derive tile sizes from the
# native resolution instead of forcing multi-thousand pixel requests.
GIBS_PIXELS_PER_DEGREE = 1024
GIBS_MIN_TILE_PIXELS = 8
GIBS_MAX_TILE_PIXELS = 4096
GIBS_DEFAULT_TIME = "default"

RECENT_LOOKBACK_DAYS = 1
FALLBACK_LOOKBACK_DAYS = 3

MIN_DIM = 0.01
MAX_DIM = 0.5
MAX_TILES_PER_RUN = 50
REQUEST_TIMEOUT = httpx.Timeout(60.0)

_CAPABILITIES_PARAMS = {
    "SERVICE": "WMS",
    "REQUEST": "GetCapabilities",
    "VERSION": "1.3.0",
}

_layer_latest_date_cache: Dict[str, str] = {}


@dataclass
class AreaTile:
    """Metadata for a satellite tile downloaded from the NASA GIBS imagery service."""

    lat: float
    lon: float
    path: Path
    source_url: str
    pixel_size: int


async def download_gibs_area_tiles(
    *,
    north: float,
    south: float,
    east: float,
    west: float,
    dim: float,
    output_dir: Path,
    date: str | None = None,
) -> Tuple[List[AreaTile], List[str]]:
    """Download a grid of imagery tiles from GIBS that cover the requested bounding box.

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

    if total_tiles > MAX_TILES_PER_RUN:
        raise ValueError(
            "Requested area requires "
            f"{total_tiles} tiles. Reduce coverage or increase the tile size to stay below "
            f"the limit of {MAX_TILES_PER_RUN} requests per scan."
        )

    tiles: List[AreaTile] = []
    failures: List[str] = []

    tile_pixels = _tile_pixel_size(dim)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        time_param = await _resolve_time_parameter(client, date)
        for lat in lat_centers:
            for lon in lon_centers:
                south, north, west_bound, east_bound = _tile_bounds(lat, lon, dim)
                bbox = f"{south:.6f},{west_bound:.6f},{north:.6f},{east_bound:.6f}"
                params = {
                    "SERVICE": "WMS",
                    "REQUEST": "GetMap",
                    "FORMAT": GIBS_IMAGE_FORMAT,
                    "VERSION": "1.3.0",
                    "STYLES": "",
                    "LAYERS": GIBS_DEFAULT_LAYER,
                    "WIDTH": tile_pixels,
                    "HEIGHT": tile_pixels,
                    "CRS": "EPSG:4326",
                    "BBOX": bbox,
                }

                if time_param:
                    params["TIME"] = time_param

                try:
                    response = await client.get(GIBS_WMS_URL, params=params)
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = _short_error_detail(exc.response.text)
                    failures.append(
                        f"lat {lat:.4f}, lon {lon:.4f}: {exc.response.status_code} {detail}"
                    )
                    logger.warning(
                        "GIBS imagery request failed with status %s: %s",
                        exc.response.status_code,
                        detail,
                    )
                    continue
                except httpx.RequestError as exc:
                    failures.append(f"lat {lat:.4f}, lon {lon:.4f}: {exc}")
                    logger.warning("GIBS imagery request error: %s", exc)
                    continue

                if not _is_image_response(response):
                    content_type = response.headers.get("Content-Type", "unknown")
                    detail = _short_error_detail(response.text)
                    failures.append(
                        f"lat {lat:.4f}, lon {lon:.4f}: unexpected payload ({content_type}): {detail}"
                    )
                    logger.warning(
                        "GIBS imagery request returned non-image payload (%s): %s",
                        content_type,
                        detail,
                    )
                    continue

                filename = _tile_filename(lat, lon, date)
                tile_path = output_dir / filename
                tile_path.write_bytes(response.content)
                pixel_size = _actual_tile_size(tile_path, tile_pixels)
                tiles.append(
                    AreaTile(
                        lat=lat,
                        lon=lon,
                        path=tile_path,
                        source_url=str(response.url),
                        pixel_size=pixel_size,
                    )
                )

    return tiles, failures


async def _resolve_time_parameter(
    client: httpx.AsyncClient, requested_date: str | None
) -> str:
    if requested_date:
        return requested_date

    cached_date = _layer_latest_date_cache.get(GIBS_DEFAULT_LAYER)
    if cached_date:
        return cached_date

    discovered_date = await _fetch_latest_available_date(client, GIBS_DEFAULT_LAYER)
    if discovered_date:
        _layer_latest_date_cache[GIBS_DEFAULT_LAYER] = discovered_date
        return discovered_date

    fallback = _fallback_date_string()
    logger.warning(
        "Falling back to %s for GIBS layer %s due to missing capabilities data.",
        fallback,
        GIBS_DEFAULT_LAYER,
    )
    return fallback


async def _fetch_latest_available_date(
    client: httpx.AsyncClient, layer: str
) -> str | None:
    try:
        response = await client.get(GIBS_WMS_URL, params=_CAPABILITIES_PARAMS)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch GIBS capabilities: %s", exc)
        return None

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        logger.warning("Failed to parse GIBS capabilities XML: %s", exc)
        return None

    dimension = _find_time_dimension(root, layer)
    if dimension is None:
        return None

    dimension_text = (dimension.text or "").strip()
    reference_date = _reference_date()

    latest = _latest_date_from_dimension(dimension_text, reference_date)
    if latest is None:
        latest = _latest_date_from_dimension(dimension_text, _current_utc_date())

    if latest is None:
        default_attr = dimension.attrib.get("default")
        default_date = _safe_parse_date(default_attr)
        if default_date:
            today = _current_utc_date()
            if default_date > today:
                default_date = today
            latest = default_date

    if latest is None:
        return None

    return latest.isoformat()


def _find_time_dimension(root: ET.Element, layer_name: str) -> ET.Element | None:
    namespaces = {"wms": "http://www.opengis.net/wms"}
    for layer in root.findall(".//wms:Layer", namespaces):
        name_element = layer.find("wms:Name", namespaces)
        if name_element is None:
            continue
        if (name_element.text or "").strip() != layer_name:
            continue
        dimension = layer.find("wms:Dimension[@name='time']", namespaces)
        if dimension is not None:
            return dimension
    return None


def _latest_date_from_dimension(dimension_text: str, limit: dt_date) -> dt_date | None:
    if not dimension_text:
        return None

    latest: dt_date | None = None
    for chunk in dimension_text.split(","):
        token = chunk.strip()
        if not token:
            continue
        parts = token.split("/")
        if len(parts) == 1:
            candidate = _safe_parse_date(parts[0])
            if candidate is None or candidate > limit:
                continue
        else:
            start = _safe_parse_date(parts[0])
            end = _safe_parse_date(parts[1]) if len(parts) > 1 else None
            if start and start > limit:
                continue
            if end is None or end > limit:
                end = limit
            if start and end < start:
                end = start
            candidate = end
        if candidate is None:
            continue
        if candidate > limit:
            candidate = limit
        if latest is None or candidate > latest:
            latest = candidate

    return latest


def _safe_parse_date(token: str | None) -> dt_date | None:
    if not token:
        return None
    token = token.strip()
    if not token:
        return None
    if len(token) >= 10:
        token = token[:10]
    try:
        return dt_date.fromisoformat(token)
    except ValueError:
        return None


def _current_utc_date() -> dt_date:
    return datetime.utcnow().date()


def _reference_date() -> dt_date:
    return _current_utc_date() - timedelta(days=RECENT_LOOKBACK_DAYS)


def _fallback_date_string() -> str:
    return (_current_utc_date() - timedelta(days=FALLBACK_LOOKBACK_DAYS)).isoformat()


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
            f"Tile size (dim) must be between {MIN_DIM} and {MAX_DIM} degrees as required by the GIBS API."
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
        return f"gibs_{date_token}_{lat_token}_{lon_token}.png"
    return f"gibs_{lat_token}_{lon_token}.png"


def _sanitize_coord(value: float) -> str:
    token = f"{value:+.4f}".replace("+", "p").replace("-", "m").replace(".", "_")
    return token


def _tile_bounds(lat: float, lon: float, dim: float) -> Tuple[float, float, float, float]:
    half_dim = dim / 2
    south = _clamp(lat - half_dim, -90.0, 90.0)
    north = _clamp(lat + half_dim, -90.0, 90.0)
    west = _clamp(lon - half_dim, -180.0, 180.0)
    east = _clamp(lon + half_dim, -180.0, 180.0)
    return south, north, west, east


def _tile_pixel_size(dim: float) -> int:
    """Derive the pixel resolution for a GIBS tile based on the requested dimension."""

    estimated_pixels = max(math.ceil(dim * GIBS_PIXELS_PER_DEGREE), GIBS_MIN_TILE_PIXELS)
    if estimated_pixels % 2:
        estimated_pixels += 1
    estimated_pixels = min(estimated_pixels, GIBS_MAX_TILE_PIXELS)
    return int(estimated_pixels)


def _short_error_detail(detail: str) -> str:
    detail = detail.strip()
    if len(detail) > 160:
        return f"{detail[:157]}..."
    return detail or "(no detail)"


def _actual_tile_size(tile_path: Path, fallback: int) -> int:
    """Inspect a downloaded tile to discover its true pixel dimension."""

    try:
        with Image.open(tile_path) as image:
            width, height = image.size
    except Exception:  # pragma: no cover - guard against corrupt downloads
        return fallback

    longest_edge = max(width or 0, height or 0)
    if longest_edge <= 0:
        return fallback
    return int(longest_edge)


def _is_image_response(response: httpx.Response) -> bool:
    content_type = response.headers.get("Content-Type", "")
    return "image" in content_type.lower()


# Backwards compatibility for older imports
download_nasa_area_tiles = download_gibs_area_tiles
