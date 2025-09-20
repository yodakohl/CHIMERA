from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import date as dt_date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import httpx
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

from ..database import DATA_DIR
from .cache import TileCache
from .usage import record_api_usage

logger = logging.getLogger(__name__)

GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GIBS_DEFAULT_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
GIBS_IMAGE_FORMAT = "image/png"
# The VIIRS "best" layer provides roughly 1 km per pixel resolution which equates to
# ~1024 pixels per degree at the equator. Requesting larger tiles simply produces
# upscaled imagery without exposing additional detail, so we derive tile sizes from the
# native resolution instead of forcing multi-thousand pixel requests. Extremely small
# bounding boxes would otherwise result in 10–50 pixel images that are unusable for the
# downstream vision models, therefore each request is clamped to a minimum of 256 pixels
# per side.
GIBS_PIXELS_PER_DEGREE = 1024
GIBS_MIN_TILE_PIXELS = 256
GIBS_MAX_TILE_PIXELS = 4096
GIBS_DEFAULT_TIME = "default"
NEIGHBOR_IDENTICAL_THRESHOLD = 0.995

# High-resolution aerial imagery from the USGS National Agriculture Imagery Program
# (NAIP). Coverage is limited to the continental United States but offers ~1 m per
# pixel orthophotography which is substantially sharper than the global NASA mosaics.
NAIP_WMS_URLS: Tuple[str, ...] = (
    "https://services.nationalmap.gov/arcgis/services/USGSNAIPPlus/MapServer/WmsServer",
    "https://services.nationalmap.gov/arcgis/services/USGSNAIPPlus/MapServer/WMSServer",
    "https://basemap.nationalmap.gov/arcgis/services/USGSNAIPPlus/MapServer/WmsServer",
)
NAIP_LAYER_NAME = "0"
NAIP_IMAGE_FORMAT = "image/jpeg"
NAIP_PIXELS_PER_DEGREE = 36000
NAIP_MIN_TILE_PIXELS = 512
NAIP_MAX_TILE_PIXELS = 4096
NAIP_COVERAGE_LAT_RANGE = (18.0, 50.0)
NAIP_COVERAGE_LON_RANGE = (-128.0, -64.0)
NAIP_COVERAGE_NOTE = (
    "outside USGS NAIP coverage (continental United States; approximately "
    f"{NAIP_COVERAGE_LAT_RANGE[0]:.1f}°–{NAIP_COVERAGE_LAT_RANGE[1]:.1f}° latitude and "
    f"{NAIP_COVERAGE_LON_RANGE[0]:.1f}°–{NAIP_COVERAGE_LON_RANGE[1]:.1f}° longitude)."
)

MAPTILER_TILE_BASE_URL = "https://api.maptiler.com/tiles"
MAPTILER_STYLE = "satellite"
MAPTILER_IMAGE_EXTENSION = "jpg"
MAPTILER_IMAGE_FORMAT = "image/jpeg"
MAPTILER_API_KEY_ENV = "MAPTILER_API_KEY"
MAPTILER_REFERER_ENV = "MAPTILER_REFERER"
MAPTILER_PIXELS_PER_DEGREE = 36000
MAPTILER_MIN_TILE_PIXELS = 512
MAPTILER_MAX_TILE_PIXELS = 2048
MAPTILER_LATITUDE_LIMIT = 85.05112878
MAPTILER_MAX_ZOOM = 20
MAPTILER_LAYER_DESCRIPTION = "MapTiler satellite basemap (global high-res)"
MAPTILER_TILESET = "satellite-v2"
MAPTILER_TILE_SIZE = 512
MAPTILER_ATTRIBUTION_TEXT = "© MapTiler"
MAPTILER_JPEG_QUALITY = 95

RECENT_LOOKBACK_DAYS = 1
FALLBACK_LOOKBACK_DAYS = 3

MIN_DIM = 0.01
MAX_DIM = 360.0
MAX_TILES_PER_RUN = 50
REQUEST_TIMEOUT = httpx.Timeout(60.0)

IMAGERY_REQUEST_DELAY_ENV = "IMAGERY_REQUEST_DELAY"
DEFAULT_REQUEST_DELAY = 5.0
IMAGERY_CACHE_DIR_ENV = "IMAGERY_CACHE_DIR"

_tile_cache: TileCache | None = None


class ImageryProviderKey(str, Enum):
    """Identifiers for the supported remote sensing providers."""

    MAPTILER_SATELLITE = "maptiler_satellite"
    NASA_GIBS = "nasa_gibs"
    USGS_NAIP = "usgs_naip"


ProviderMetadata = Dict[str, str]


PROVIDER_METADATA: Dict[ImageryProviderKey, ProviderMetadata] = {
    ImageryProviderKey.MAPTILER_SATELLITE: {
        "label": "MapTiler Satellite (global high-res)",
        "description": (
            "Global satellite basemap from MapTiler. Requires the MAPTILER_API_KEY environment "
            "variable to be configured."
        ),
    },
    ImageryProviderKey.NASA_GIBS: {
        "label": "NASA GIBS (global satellite mosaics)",
        "description": "Worldwide coverage using VIIRS imagery with automatic Landsat WELD fallback (~30 m).",
    },
    ImageryProviderKey.USGS_NAIP: {
        "label": "USGS NAIP Plus (US aerial ~1 m)",
        "description": "High-resolution orthophotos for the continental United States from the National Map.",
    },
}

_CAPABILITIES_PARAMS = {
    "SERVICE": "WMS",
    "REQUEST": "GetCapabilities",
    "VERSION": "1.3.0",
}

@dataclass(frozen=True)
class GibsLayerConfig:
    """Configuration describing the characteristics of a NASA GIBS imagery layer."""

    name: str
    pixels_per_degree: int
    min_tile_pixels: int = GIBS_MIN_TILE_PIXELS
    max_tile_pixels: int = GIBS_MAX_TILE_PIXELS
    description: str | None = None


DEFAULT_LAYER = GibsLayerConfig(
    name=GIBS_DEFAULT_LAYER,
    pixels_per_degree=GIBS_PIXELS_PER_DEGREE,
    description="Suomi NPP VIIRS true color (~1 km)",
)

# Landsat WELD mosaics provide ~30 m per pixel coverage that is far more suitable for
# object recognition models when recent imagery is not required.
HIGH_RES_LAYER = GibsLayerConfig(
    name="Landsat_WELD_CorrectedReflectance_TrueColor_Global_Monthly",
    pixels_per_degree=3600,
    description="Landsat WELD true color mosaic (~30 m)",
)

LAYER_SEQUENCE: Tuple[GibsLayerConfig, ...] = (DEFAULT_LAYER, HIGH_RES_LAYER)

_layer_latest_date_cache: Dict[str, str] = {}


def _determine_cache_dir() -> Path:
    override = os.getenv(IMAGERY_CACHE_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return DATA_DIR / "tile_cache"


def _get_tile_cache() -> TileCache:
    global _tile_cache
    cache_dir = _determine_cache_dir()
    if _tile_cache is None or _tile_cache.root != cache_dir:
        _tile_cache = TileCache(cache_dir)
    return _tile_cache


def _request_delay_seconds() -> float:
    raw_value = os.getenv(IMAGERY_REQUEST_DELAY_ENV, "").strip()
    if not raw_value:
        return DEFAULT_REQUEST_DELAY
    try:
        delay = float(raw_value)
    except ValueError:
        return DEFAULT_REQUEST_DELAY
    return max(0.0, delay)


async def _respect_rate_limit() -> None:
    delay = _request_delay_seconds()
    if delay > 0:
        await asyncio.sleep(delay)


def _cache_key(provider: str, *parts: object) -> str:
    tokens: List[str] = [provider]
    for part in parts:
        if isinstance(part, float):
            tokens.append(f"{part:.6f}")
        elif part is None:
            tokens.append("none")
        else:
            tokens.append(str(part))
    return "|".join(tokens)


def _area_tile_from_metadata(
    *,
    lat: float,
    lon: float,
    path: Path,
    metadata: Dict[str, Any],
    default_layer: str,
    default_provider: ImageryProviderKey,
    default_label: str | None,
    default_degree_size: float,
    fallback_pixels: int,
    default_native_dim: float,
    default_pixels_per_degree: float,
) -> AreaTile:
    provider = str(metadata.get("provider") or default_provider.value)
    provider_label = metadata.get("provider_label") or default_label
    layer = str(metadata.get("layer") or default_layer)
    source_url = str(metadata.get("source_url") or "")

    degree_size = float(metadata.get("degree_size") or 0.0)
    if degree_size <= 0:
        degree_size = default_degree_size

    native_dim = float(metadata.get("native_dim") or 0.0)
    if native_dim <= 0:
        native_dim = default_native_dim

    pixel_size = int(metadata.get("pixel_size") or 0)
    if pixel_size <= 0:
        pixel_size = _actual_tile_size(path, fallback_pixels)

    pixels_per_degree = float(metadata.get("pixels_per_degree") or 0.0)
    if pixels_per_degree <= 0 and degree_size > 0:
        pixels_per_degree = pixel_size / degree_size
    elif pixels_per_degree <= 0:
        pixels_per_degree = default_pixels_per_degree

    layer_description = metadata.get("layer_description")

    return AreaTile(
        lat=lat,
        lon=lon,
        path=path,
        source_url=source_url,
        pixel_size=pixel_size,
        degree_size=degree_size,
        layer=layer,
        native_dim=native_dim,
        pixels_per_degree=pixels_per_degree,
        layer_description=layer_description,
        provider=provider,
        provider_label=provider_label,
    )


@dataclass
class AreaTile:
    """Metadata describing an imagery tile retrieved from a remote provider."""

    lat: float
    lon: float
    path: Path
    source_url: str
    pixel_size: int
    degree_size: float
    layer: str
    native_dim: float
    pixels_per_degree: float
    low_detail: bool = field(default=False)
    detail_ratio: float | None = field(default=None)
    layer_description: str | None = field(default=None)
    provider: str = field(default=ImageryProviderKey.NASA_GIBS.value)
    provider_label: str | None = field(default=None)


async def download_area_tiles(
    *,
    provider: ImageryProviderKey,
    north: float,
    south: float,
    east: float,
    west: float,
    dim: float,
    output_dir: Path,
    date: str | None = None,
) -> Tuple[List[AreaTile], List[str]]:
    """Dispatch imagery downloads to the selected remote sensing provider."""

    if isinstance(provider, str):  # pragma: no cover - defensive for direct calls
        provider = ImageryProviderKey(provider)

    if provider == ImageryProviderKey.MAPTILER_SATELLITE:
        return await download_maptiler_area_tiles(
            north=north,
            south=south,
            east=east,
            west=west,
            dim=dim,
            output_dir=output_dir,
            date=date,
        )
    if provider == ImageryProviderKey.NASA_GIBS:
        return await download_gibs_area_tiles(
            north=north,
            south=south,
            east=east,
            west=west,
            dim=dim,
            output_dir=output_dir,
            date=date,
        )
    if provider == ImageryProviderKey.USGS_NAIP:
        return await download_naip_area_tiles(
            north=north,
            south=south,
            east=east,
            west=west,
            dim=dim,
            output_dir=output_dir,
            date=date,
        )

    raise ValueError(f"Unsupported imagery provider: {provider}")


async def download_maptiler_area_tiles(
    *,
    north: float,
    south: float,
    east: float,
    west: float,
    dim: float,
    output_dir: Path,
    date: str | None = None,
) -> Tuple[List[AreaTile], List[str]]:
    """Download satellite tiles from the MapTiler static imagery API."""

    _validate_bounds(north=north, south=south, east=east, west=west)
    _validate_dim(dim)

    api_key = os.getenv(MAPTILER_API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(
            "MapTiler imagery requires the MAPTILER_API_KEY environment variable to be set."
        )

    referer = os.getenv(MAPTILER_REFERER_ENV, "").strip()
    request_headers: Dict[str, str] | None = None
    if referer:
        request_headers = {"Referer": referer}

    output_dir.mkdir(parents=True, exist_ok=True)

    lat_limit = MAPTILER_LATITUDE_LIMIT
    limit_epsilon = 1e-6
    if north > lat_limit + limit_epsilon or south < -lat_limit - limit_epsilon:
        raise ValueError(
            "MapTiler satellite imagery covers latitudes between "
            f"{-lat_limit:.4f}° and {lat_limit:.4f}°. "
            "Please adjust the bounding box or switch to the NASA GIBS provider for polar regions."
        )

    clamped_south = _clamp(south, -lat_limit, lat_limit)
    clamped_north = _clamp(north, -lat_limit, lat_limit)
    clamp_min = _clamp(-lat_limit + dim / 2, -lat_limit, lat_limit)
    clamp_max = _clamp(lat_limit - dim / 2, -lat_limit, lat_limit)
    if clamp_min >= clamp_max:
        clamp_min, clamp_max = -lat_limit, lat_limit

    lat_centers = _build_axis_centers(
        minimum=clamped_south,
        maximum=clamped_north,
        dim=dim,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
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

    provider_label = PROVIDER_METADATA[ImageryProviderKey.MAPTILER_SATELLITE]["label"]
    tiles: List[AreaTile] = []
    failures: List[str] = []
    native_dim = _maptiler_native_dim()

    cache = _get_tile_cache()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for lat in lat_centers:
            for lon in lon_centers:
                lat = _clamp(lat, clamp_min, clamp_max)
                south_bound, north_bound, west_bound, east_bound = _tile_bounds(lat, lon, dim)
                south_bound = _clamp(south_bound, -lat_limit, lat_limit)
                north_bound = _clamp(north_bound, -lat_limit, lat_limit)
                degree_span = max(dim, north_bound - south_bound, east_bound - west_bound, MIN_DIM)
                tile_pixels = _maptiler_tile_pixels(degree_span)
                zoom = _maptiler_zoom_level(
                    north=north_bound,
                    south=south_bound,
                    east=east_bound,
                    west=west_bound,
                    width=tile_pixels,
                    height=tile_pixels,
                )

                filename = _tile_filename(
                    lat,
                    lon,
                    date,
                    prefix="maptiler",
                    extension=MAPTILER_IMAGE_EXTENSION,
                )
                tile_path = output_dir / filename
                cache_key = _cache_key(
                    ImageryProviderKey.MAPTILER_SATELLITE.value,
                    lat,
                    lon,
                    degree_span,
                    zoom,
                    tile_pixels,
                    date or "",
                )

                cached_metadata = cache.load(cache_key, MAPTILER_IMAGE_EXTENSION, tile_path)
                if cached_metadata is not None:
                    tiles.append(
                        _area_tile_from_metadata(
                            lat=lat,
                            lon=lon,
                            path=tile_path,
                            metadata=cached_metadata,
                            default_layer=MAPTILER_STYLE,
                            default_provider=ImageryProviderKey.MAPTILER_SATELLITE,
                            default_label=provider_label,
                            default_degree_size=degree_span,
                            fallback_pixels=tile_pixels,
                            default_native_dim=native_dim,
                            default_pixels_per_degree=MAPTILER_PIXELS_PER_DEGREE,
                        )
                    )
                    continue

                await _respect_rate_limit()
                try:
                    (
                        image_bytes,
                        rendered_pixels,
                        source_detail,
                        request_count,
                    ) = await _maptiler_render_tile_image(
                        client=client,
                        api_key=api_key,
                        headers=request_headers,
                        zoom=zoom,
                        north=north_bound,
                        south=south_bound,
                        east=east_bound,
                        west=west_bound,
                        target_pixels=tile_pixels,
                    )
                except MapTilerDownloadError as exc:
                    detail = str(exc)
                    failures.append(f"lat {lat:.4f}, lon {lon:.4f}: {detail}")
                    logger.warning("MapTiler imagery request failed: %s", detail)
                    continue

                tile_path.write_bytes(image_bytes)

                pixel_size = _actual_tile_size(tile_path, rendered_pixels)
                pixels_per_degree = pixel_size / degree_span if degree_span > 0 else 0.0

                source_url = source_detail or _maptiler_fallback_source_summary(
                    zoom=zoom,
                    west=west_bound,
                    east=east_bound,
                    south=south_bound,
                    north=north_bound,
                )

                metadata = {
                    "provider": ImageryProviderKey.MAPTILER_SATELLITE.value,
                    "provider_label": provider_label,
                    "pixel_size": pixel_size,
                    "degree_size": degree_span,
                    "source_url": source_url,
                    "layer": MAPTILER_STYLE,
                    "native_dim": native_dim,
                    "pixels_per_degree": pixels_per_degree or MAPTILER_PIXELS_PER_DEGREE,
                    "layer_description": MAPTILER_LAYER_DESCRIPTION,
                }
                cache.store(cache_key, MAPTILER_IMAGE_EXTENSION, image_bytes, metadata)
                record_api_usage(
                    ImageryProviderKey.MAPTILER_SATELLITE.value,
                    increment=max(1, int(request_count)),
                )

                tiles.append(
                    _area_tile_from_metadata(
                        lat=lat,
                        lon=lon,
                        path=tile_path,
                        metadata=metadata,
                        default_layer=MAPTILER_STYLE,
                        default_provider=ImageryProviderKey.MAPTILER_SATELLITE,
                        default_label=provider_label,
                        default_degree_size=degree_span,
                        fallback_pixels=rendered_pixels,
                        default_native_dim=native_dim,
                        default_pixels_per_degree=MAPTILER_PIXELS_PER_DEGREE,
                    )
                )

    return tiles, failures


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

    The downloader automatically expands the tile size when the requested
    dimension would produce blocky upscaled imagery so the downstream models
    receive higher fidelity inputs. Returns a tuple consisting of successfully
    downloaded tiles and a list of human-readable error messages for tiles that
    could not be fetched.
    """

    _validate_bounds(north=north, south=south, east=east, west=west)
    _validate_dim(dim)

    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated_failures: List[str] = []
    best_attempt: Tuple[GibsLayerConfig, str | None, float] | None = None
    tiles: List[AreaTile] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for index, layer in enumerate(LAYER_SEQUENCE):
            time_param = await _resolve_time_parameter(client, date, layer)
            attempt_tiles, attempt_failures, used_dim, low_detail_remaining, avg_ratio = await _download_with_adaptive_dim(
                client=client,
                layer=layer,
                time_param=time_param,
                north=north,
                south=south,
                east=east,
                west=west,
                initial_dim=dim,
                output_dir=output_dir,
                date=date,
            )

            aggregated_failures.extend(attempt_failures)

            if attempt_tiles:
                tiles = attempt_tiles
                best_attempt = (layer, time_param, used_dim)
                if avg_ratio is not None:
                    logger.debug(
                        "Average neighbor similarity for layer %s at %.3f° tiles: %.3f",
                        layer.name,
                        used_dim,
                        avg_ratio,
                    )
                if not low_detail_remaining:
                    return tiles, aggregated_failures

                next_layer = LAYER_SEQUENCE[index + 1] if index + 1 < len(LAYER_SEQUENCE) else None
                if next_layer:
                    logger.info(
                        "Imagery from %s remains low detail; attempting higher resolution layer %s.",
                        layer.name,
                        next_layer.name,
                    )
                    continue

                return tiles, aggregated_failures

        # If we reach this point no layer produced acceptable detail. Re-download the
        # best attempt so the output directory contains the final imagery.
        if best_attempt is not None:
            layer, time_param, used_dim = best_attempt
            _clear_output_directory(output_dir)
            tiles, retry_failures = await _download_tiles_for_dim(
                client=client,
                layer=layer,
                time_param=time_param,
                north=north,
                south=south,
                east=east,
                west=west,
                dim=used_dim,
                output_dir=output_dir,
                date=date,
            )
            aggregated_failures.extend(retry_failures)
            if tiles:
                _, detail_flags, detail_scores = _tiles_need_higher_resolution(tiles)
                for tile, flag, score in zip(tiles, detail_flags, detail_scores):
                    tile.low_detail = flag
                    tile.detail_ratio = score

    return tiles, aggregated_failures


async def download_naip_area_tiles(
    *,
    north: float,
    south: float,
    east: float,
    west: float,
    dim: float,
    output_dir: Path,
    date: str | None = None,
) -> Tuple[List[AreaTile], List[str]]:
    """Download high-resolution aerial tiles from the USGS NAIP WMS service."""

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
    native_dim = max(MIN_DIM, NAIP_MIN_TILE_PIXELS / NAIP_PIXELS_PER_DEGREE)
    provider_label = PROVIDER_METADATA[ImageryProviderKey.USGS_NAIP]["label"]
    cache = _get_tile_cache()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for lat in lat_centers:
            for lon in lon_centers:
                south_bound, north_bound, west_bound, east_bound = _tile_bounds(lat, lon, dim)
                lat_span = max(north_bound - south_bound, MIN_DIM)
                lon_span = max(east_bound - west_bound, MIN_DIM)
                width = _naip_tile_pixels(lon_span)
                height = _naip_tile_pixels(lat_span)

                if not _naip_tile_intersects_coverage(
                    south_bound, north_bound, west_bound, east_bound
                ):
                    failures.append(
                        f"lat {lat:.4f}, lon {lon:.4f}: {NAIP_COVERAGE_NOTE}"
                    )
                    logger.info(
                        "Skipping NAIP tile at lat %.4f lon %.4f because it falls outside the service coverage "
                        "(lat %.1f°–%.1f°, lon %.1f°–%.1f°).",
                        lat,
                        lon,
                        NAIP_COVERAGE_LAT_RANGE[0],
                        NAIP_COVERAGE_LAT_RANGE[1],
                        NAIP_COVERAGE_LON_RANGE[0],
                        NAIP_COVERAGE_LON_RANGE[1],
                    )
                    continue

                bbox_lon_lat = (
                    f"{west_bound:.6f},{south_bound:.6f},{east_bound:.6f},{north_bound:.6f}"
                )
                bbox_lat_lon = (
                    f"{south_bound:.6f},{west_bound:.6f},{north_bound:.6f},{east_bound:.6f}"
                )

                base_params = {
                    "SERVICE": "WMS",
                    "REQUEST": "GetMap",
                    "FORMAT": NAIP_IMAGE_FORMAT,
                    "STYLES": "",
                    "LAYERS": NAIP_LAYER_NAME,
                    "WIDTH": width,
                    "HEIGHT": height,
                }

                degree_span = max(dim, lat_span, lon_span)
                filename = _tile_filename(
                    lat,
                    lon,
                    date,
                    prefix="naip",
                    extension=".jpg",
                )
                tile_path = output_dir / filename
                cache_key = _cache_key(
                    ImageryProviderKey.USGS_NAIP.value,
                    lat,
                    lon,
                    degree_span,
                    width,
                    height,
                    date or "",
                )

                cached_metadata = cache.load(cache_key, ".jpg", tile_path)
                if cached_metadata is not None:
                    tiles.append(
                        _area_tile_from_metadata(
                            lat=lat,
                            lon=lon,
                            path=tile_path,
                            metadata=cached_metadata,
                            default_layer="USGSNAIPPlus",
                            default_provider=ImageryProviderKey.USGS_NAIP,
                            default_label=provider_label,
                            default_degree_size=degree_span,
                            fallback_pixels=max(width, height),
                            default_native_dim=native_dim,
                            default_pixels_per_degree=NAIP_PIXELS_PER_DEGREE,
                        )
                    )
                    continue

                await _respect_rate_limit()

                request_attempts = [
                    # WMS 1.3.0 swaps the axis order for EPSG:4326 compared to the legacy
                    # specification, however some ArcGIS deployments still honour the older
                    # behaviour. Try the standards-compliant latitude/longitude order first,
                    # then fall back to the alternate ordering before attempting version 1.1.1.
                    ("1.3.0", "CRS", bbox_lat_lon, "lat-lon"),
                    ("1.3.0", "CRS", bbox_lon_lat, "lon-lat"),
                    # A number of NAIP endpoints erroneously expect latitude/longitude even
                    # when negotiating WMS 1.1.1 responses. Include both permutations so the
                    # downloader can gracefully recover instead of failing with a 400 error.
                    ("1.1.1", "SRS", bbox_lon_lat, "lon-lat"),
                    ("1.1.1", "SRS", bbox_lat_lon, "lat-lon"),
                ]

                response: httpx.Response | None = None
                last_error: str | None = None
                success_attempt: tuple[str, str] | None = None

                for version, crs_key, bbox_value, axis_label in request_attempts:
                    params = dict(base_params)
                    params["VERSION"] = version
                    params["BBOX"] = bbox_value
                    params[crs_key] = "EPSG:4326"

                    for endpoint in NAIP_WMS_URLS:
                        try:
                            candidate = await client.get(endpoint, params=params)
                            candidate.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            detail = _short_error_detail(exc.response.text)
                            last_error = (
                                f"{exc.response.status_code} {detail} "
                                f"(endpoint {endpoint}, version {version}, axis {axis_label})"
                            )
                            logger.warning(
                                "USGS NAIP imagery request failed with status %s from %s (version %s, %s axis order): %s",
                                exc.response.status_code,
                                endpoint,
                                version,
                                axis_label,
                                detail,
                            )
                            continue
                        except httpx.RequestError as exc:
                            last_error = (
                                f"{exc} (endpoint {endpoint}, version {version}, axis {axis_label})"
                            )
                            logger.warning(
                                "USGS NAIP imagery request error from %s (version %s, %s axis order): %s",
                                endpoint,
                                version,
                                axis_label,
                                exc,
                            )
                            continue

                        if not _is_image_response(candidate):
                            content_type = candidate.headers.get("Content-Type", "unknown")
                            detail = _short_error_detail(candidate.text)
                            last_error = (
                                f"unexpected payload ({content_type}) from {endpoint}, version {version}, "
                                f"axis {axis_label}: {detail}"
                            )
                            logger.warning(
                                "USGS NAIP imagery request returned non-image payload (%s) from %s (version %s, %s axis order): %s",
                                content_type,
                                endpoint,
                                version,
                                axis_label,
                                detail,
                            )
                            continue

                        response = candidate
                        success_attempt = (version, axis_label)
                        if endpoint != NAIP_WMS_URLS[0]:
                            logger.info(
                                "USGS NAIP imagery request succeeded using fallback endpoint %s (version %s, %s axis order)",
                                endpoint,
                                version,
                                axis_label,
                            )
                        break

                    if response is not None:
                        if success_attempt != ("1.3.0", "lon-lat"):
                            logger.info(
                                "USGS NAIP imagery request succeeded using alternate configuration (version %s, %s axis order)",
                                version,
                                axis_label,
                            )
                        break

                if response is None:
                    error_message = last_error or "no NAIP endpoints available"
                    failures.append(f"lat {lat:.4f}, lon {lon:.4f}: {error_message}")
                    continue

                tile_path.write_bytes(response.content)

                pixel_size = _actual_tile_size(tile_path, max(width, height))
                lon_resolution = width / lon_span if lon_span > 0 else 0.0
                lat_resolution = height / lat_span if lat_span > 0 else 0.0
                pixels_per_degree = max(lon_resolution, lat_resolution, 0.0)

                metadata = {
                    "provider": ImageryProviderKey.USGS_NAIP.value,
                    "provider_label": provider_label,
                    "pixel_size": pixel_size,
                    "degree_size": degree_span,
                    "source_url": str(response.url),
                    "layer": "USGSNAIPPlus",
                    "native_dim": native_dim,
                    "pixels_per_degree": pixels_per_degree or NAIP_PIXELS_PER_DEGREE,
                    "layer_description": "USGS NAIP Plus aerial imagery (~1 m)",
                }
                cache.store(cache_key, ".jpg", response.content, metadata)
                record_api_usage(ImageryProviderKey.USGS_NAIP.value, increment=1)

                tiles.append(
                    _area_tile_from_metadata(
                        lat=lat,
                        lon=lon,
                        path=tile_path,
                        metadata=metadata,
                        default_layer="USGSNAIPPlus",
                        default_provider=ImageryProviderKey.USGS_NAIP,
                        default_label=provider_label,
                        default_degree_size=degree_span,
                        fallback_pixels=max(width, height),
                        default_native_dim=native_dim,
                        default_pixels_per_degree=pixels_per_degree or NAIP_PIXELS_PER_DEGREE,
                    )
                )

    return tiles, failures


async def _download_with_adaptive_dim(
    *,
    client: httpx.AsyncClient,
    layer: GibsLayerConfig,
    time_param: str | None,
    north: float,
    south: float,
    east: float,
    west: float,
    initial_dim: float,
    output_dir: Path,
    date: str | None,
) -> Tuple[List[AreaTile], List[str], float, bool, float | None]:
    requested_dim = initial_dim
    native_dim = _minimum_native_tile_dim(layer)
    dim = max(initial_dim, native_dim)
    if dim > requested_dim + 1e-9:
        logger.info(
            "Requested tile size %.3f° is below the native resolution of %s (~%.3f°); using %.3f° tiles instead.",
            requested_dim,
            layer.name,
            native_dim,
            dim,
        )

    current_dim = dim
    tiles: List[AreaTile] = []
    failures: List[str] = []
    avg_ratio: float | None = None

    while True:
        _clear_output_directory(output_dir)
        tiles, failures = await _download_tiles_for_dim(
            client=client,
            layer=layer,
            time_param=time_param,
            north=north,
            south=south,
            east=east,
            west=west,
            dim=current_dim,
            output_dir=output_dir,
            date=date,
        )

        if not tiles:
            return tiles, failures, current_dim, False, avg_ratio

        needs_more_detail, detail_flags, detail_scores = _tiles_need_higher_resolution(tiles)
        avg_ratio = None
        if detail_scores:
            avg_ratio = sum(detail_scores) / len(detail_scores)
        for tile, flag, score in zip(tiles, detail_flags, detail_scores):
            tile.low_detail = flag
            tile.detail_ratio = score

        if not needs_more_detail:
            return tiles, failures, current_dim, False, avg_ratio

        if current_dim >= MAX_DIM - 1e-9:
            logger.info(
                "Tiles from layer %s remain low detail after reaching the maximum tile size of %.3f°.",
                layer.name,
                current_dim,
            )
            return tiles, failures, current_dim, True, avg_ratio

        new_dim = min(MAX_DIM, current_dim * 2)
        if new_dim <= current_dim + 1e-9:
            return tiles, failures, current_dim, True, avg_ratio

        ratio_text = f"{avg_ratio:.3f}" if avg_ratio is not None else "unknown"
        logger.info(
            "Tiles downloaded from layer %s at %.3f° appear low detail (average neighbor similarity %s). Retrying with %.3f° tiles.",
            layer.name,
            current_dim,
            ratio_text,
            new_dim,
        )
        current_dim = new_dim


async def _download_tiles_for_dim(
    *,
    client: httpx.AsyncClient,
    layer: GibsLayerConfig,
    time_param: str | None,
    north: float,
    south: float,
    east: float,
    west: float,
    dim: float,
    output_dir: Path,
    date: str | None,
) -> Tuple[List[AreaTile], List[str]]:
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
    tile_pixels = _tile_pixel_size(dim, layer)
    cache = _get_tile_cache()
    provider_label = PROVIDER_METADATA[ImageryProviderKey.NASA_GIBS]["label"]
    native_dim = _minimum_native_tile_dim(layer)

    for lat in lat_centers:
        for lon in lon_centers:
            south_bound, north_bound, west_bound, east_bound = _tile_bounds(lat, lon, dim)
            bbox = f"{south_bound:.6f},{west_bound:.6f},{north_bound:.6f},{east_bound:.6f}"
            params = {
                "SERVICE": "WMS",
                "REQUEST": "GetMap",
                "FORMAT": GIBS_IMAGE_FORMAT,
                "VERSION": "1.3.0",
                "STYLES": "",
                "LAYERS": layer.name,
                "WIDTH": tile_pixels,
                "HEIGHT": tile_pixels,
                "CRS": "EPSG:4326",
                "BBOX": bbox,
            }

            if time_param:
                params["TIME"] = time_param

            filename = _tile_filename(lat, lon, date)
            tile_path = output_dir / filename
            cache_key = _cache_key(
                ImageryProviderKey.NASA_GIBS.value,
                layer.name,
                lat,
                lon,
                dim,
                tile_pixels,
                time_param or "",
                date or "",
            )

            cached_metadata = cache.load(cache_key, ".png", tile_path)
            if cached_metadata is not None:
                tiles.append(
                    _area_tile_from_metadata(
                        lat=lat,
                        lon=lon,
                        path=tile_path,
                        metadata=cached_metadata,
                        default_layer=layer.name,
                        default_provider=ImageryProviderKey.NASA_GIBS,
                        default_label=provider_label,
                        default_degree_size=dim,
                        fallback_pixels=tile_pixels,
                        default_native_dim=native_dim,
                        default_pixels_per_degree=layer.pixels_per_degree,
                    )
                )
                continue

            await _respect_rate_limit()

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

            tile_path.write_bytes(response.content)
            pixel_size = _actual_tile_size(tile_path, tile_pixels)
            metadata = {
                "provider": ImageryProviderKey.NASA_GIBS.value,
                "provider_label": provider_label,
                "pixel_size": pixel_size,
                "degree_size": dim,
                "source_url": str(response.url),
                "layer": layer.name,
                "native_dim": native_dim,
                "pixels_per_degree": layer.pixels_per_degree,
                "layer_description": layer.description,
                "time_param": time_param,
            }
            cache.store(cache_key, ".png", response.content, metadata)
            record_api_usage(ImageryProviderKey.NASA_GIBS.value, increment=1)
            tiles.append(
                _area_tile_from_metadata(
                    lat=lat,
                    lon=lon,
                    path=tile_path,
                    metadata=metadata,
                    default_layer=layer.name,
                    default_provider=ImageryProviderKey.NASA_GIBS,
                    default_label=provider_label,
                    default_degree_size=dim,
                    fallback_pixels=tile_pixels,
                    default_native_dim=native_dim,
                    default_pixels_per_degree=layer.pixels_per_degree,
                )
            )

    return tiles, failures


async def _resolve_time_parameter(
    client: httpx.AsyncClient, requested_date: str | None, layer: GibsLayerConfig
) -> str:
    if requested_date:
        return requested_date

    cached_date = _layer_latest_date_cache.get(layer.name)
    if cached_date:
        return cached_date

    discovered_date = await _fetch_latest_available_date(client, layer.name)
    if discovered_date:
        _layer_latest_date_cache[layer.name] = discovered_date
        return discovered_date

    if layer.name == GIBS_DEFAULT_LAYER:
        fallback = _fallback_date_string()
        logger.warning(
            "Falling back to %s for GIBS layer %s due to missing capabilities data.",
            fallback,
            layer.name,
        )
        return fallback

    logger.info(
        "Layer %s does not advertise a usable time dimension; requesting default imagery without a TIME parameter.",
        layer.name,
    )
    return None


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
            f"Tile size (dim) must be between {MIN_DIM} and {MAX_DIM} degrees for the supported imagery services."
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


def _tile_filename(
    lat: float,
    lon: float,
    date: str | None,
    *,
    prefix: str = "gibs",
    extension: str = ".png",
) -> str:
    lat_token = _sanitize_coord(lat)
    lon_token = _sanitize_coord(lon)
    suffix = extension if extension.startswith(".") else f".{extension}"
    if date:
        date_token = date.replace("-", "")
        return f"{prefix}_{date_token}_{lat_token}_{lon_token}{suffix}"
    return f"{prefix}_{lat_token}_{lon_token}{suffix}"


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


def _tiles_need_higher_resolution(
    tiles: Sequence[AreaTile],
) -> Tuple[bool, List[bool], List[float]]:
    if not tiles:
        return False, [], []

    low_detail_flags: List[bool] = []
    detail_scores: List[float] = []
    low_detail_count = 0
    for tile in tiles:
        ratio = _tile_neighbor_similarity_ratio(tile.path)
        detail_scores.append(ratio)
        is_low_detail = ratio >= NEIGHBOR_IDENTICAL_THRESHOLD
        low_detail_flags.append(is_low_detail)
        if is_low_detail:
            low_detail_count += 1

    threshold = max(1, len(tiles) // 2)
    needs_more_detail = low_detail_count >= threshold
    return needs_more_detail, low_detail_flags, detail_scores


def _tile_neighbor_similarity_ratio(tile_path: Path) -> float:
    try:
        with Image.open(tile_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if width < 2 or height < 2:
                return 1.0

            pixels = image.load()
            identical = 0
            comparisons = 0

            for y in range(height):
                for x in range(width):
                    if x > 0:
                        comparisons += 1
                        if pixels[x, y] == pixels[x - 1, y]:
                            identical += 1
                    if y > 0:
                        comparisons += 1
                        if pixels[x, y] == pixels[x, y - 1]:
                            identical += 1

            if comparisons == 0:
                return 1.0

            return identical / comparisons
    except Exception:
        return 0.0


def _clear_output_directory(output_dir: Path) -> None:
    try:
        entries = list(output_dir.iterdir())
    except FileNotFoundError:
        return

    for entry in entries:
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError:
            logger.warning("Failed to remove temporary imagery file %s", entry)


def _minimum_native_tile_dim(layer: GibsLayerConfig) -> float:
    native_min = layer.min_tile_pixels / layer.pixels_per_degree
    return max(MIN_DIM, native_min)


def _tile_pixel_size(dim: float, layer: GibsLayerConfig) -> int:
    """Derive the pixel resolution for a GIBS tile based on the requested dimension.

    The computation respects the native resolution of the layer while ensuring that the
    resulting imagery is never smaller than ``GIBS_MIN_TILE_PIXELS`` so the downstream
    analysis models receive sufficiently detailed inputs.
    """

    estimated_pixels = max(
        math.ceil(dim * layer.pixels_per_degree), layer.min_tile_pixels
    )
    if estimated_pixels % 2:
        estimated_pixels += 1
    estimated_pixels = min(estimated_pixels, layer.max_tile_pixels)
    return int(estimated_pixels)


def _maptiler_tile_pixels(degree_span: float) -> int:
    estimated = max(int(round(degree_span * MAPTILER_PIXELS_PER_DEGREE)), MAPTILER_MIN_TILE_PIXELS)
    if estimated % 2:
        estimated += 1
    return min(estimated, MAPTILER_MAX_TILE_PIXELS)


class MapTilerDownloadError(Exception):
    """Raised when the MapTiler tile imagery cannot be downloaded."""


async def _maptiler_render_tile_image(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    headers: Dict[str, str] | None,
    zoom: int,
    north: float,
    south: float,
    east: float,
    west: float,
    target_pixels: int,
) -> Tuple[bytes, int, str]:
    mosaic, urls = await _maptiler_download_tile_mosaic(
        client=client,
        api_key=api_key,
        headers=headers,
        zoom=zoom,
        north=north,
        south=south,
        east=east,
        west=west,
    )

    if mosaic.width == 0 or mosaic.height == 0:
        raise MapTilerDownloadError("received empty MapTiler mosaic")

    pixels = max(1, target_pixels)
    if mosaic.size != (pixels, pixels):
        mosaic = mosaic.resize((pixels, pixels), Image.LANCZOS)

    mosaic = _apply_maptiler_attribution(mosaic)

    buffer = io.BytesIO()
    mosaic.save(buffer, format="JPEG", quality=MAPTILER_JPEG_QUALITY)
    request_count = max(1, len(urls))
    return buffer.getvalue(), mosaic.size[0], _summarize_maptiler_urls(urls), request_count


async def _maptiler_download_tile_mosaic(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    headers: Dict[str, str] | None,
    zoom: int,
    north: float,
    south: float,
    east: float,
    west: float,
) -> Tuple[Image.Image, List[str]]:
    tile_x_west = _maptiler_lon_to_tile_x(west, zoom)
    tile_x_east = _maptiler_lon_to_tile_x(east, zoom)
    tile_y_north = _maptiler_lat_to_tile_y(north, zoom)
    tile_y_south = _maptiler_lat_to_tile_y(south, zoom)

    x_start_frac, x_end_frac, x_min, x_max = _maptiler_tile_span(tile_x_west, tile_x_east, zoom)
    y_start_frac, y_end_frac, y_min, y_max = _maptiler_tile_span(tile_y_north, tile_y_south, zoom)

    columns = x_max - x_min + 1
    rows = y_max - y_min + 1
    if columns <= 0 or rows <= 0:
        raise MapTilerDownloadError("invalid MapTiler tile coverage (empty mosaic)")

    mosaic = Image.new("RGB", (columns * MAPTILER_TILE_SIZE, rows * MAPTILER_TILE_SIZE))
    requested_urls: List[str] = []

    for row, tile_y in enumerate(range(y_min, y_max + 1)):
        for col, tile_x in enumerate(range(x_min, x_max + 1)):
            url = _maptiler_tile_url(zoom=zoom, x=tile_x, y=tile_y)
            tile_image, resolved_url = await _maptiler_request_tile(
                client=client,
                url=url,
                api_key=api_key,
                headers=headers,
            )
            mosaic.paste(tile_image, (col * MAPTILER_TILE_SIZE, row * MAPTILER_TILE_SIZE))
            requested_urls.append(resolved_url)

    x0 = max(0, min(mosaic.width, int(round((x_start_frac - x_min) * MAPTILER_TILE_SIZE))))
    x1 = max(x0 + 1, min(mosaic.width, int(round((x_end_frac - x_min) * MAPTILER_TILE_SIZE))))
    y0 = max(0, min(mosaic.height, int(round((y_start_frac - y_min) * MAPTILER_TILE_SIZE))))
    y1 = max(y0 + 1, min(mosaic.height, int(round((y_end_frac - y_min) * MAPTILER_TILE_SIZE))))

    cropped = mosaic.crop((x0, y0, x1, y1))
    return cropped, requested_urls


async def _maptiler_request_tile(
    *,
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    headers: Dict[str, str] | None,
) -> Tuple[Image.Image, str]:
    params = {"key": api_key}
    try:
        response = await client.get(url, params=params, headers=headers)
    except httpx.RequestError as exc:  # pragma: no cover - network failure path
        raise MapTilerDownloadError(str(exc)) from exc

    if response.status_code == 204:
        raise MapTilerDownloadError("no imagery available (204 No Content)")

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _maptiler_http_error_detail(exc.response)
        raise MapTilerDownloadError(f"{exc.response.status_code} {detail}") from exc

    if not _is_image_response(response):
        content_type = response.headers.get("Content-Type", "unknown")
        detail = _maptiler_http_error_detail(response)
        raise MapTilerDownloadError(f"unexpected payload ({content_type}): {detail}")

    try:
        image = Image.open(io.BytesIO(response.content))
        image.load()
    except Exception as exc:  # pragma: no cover - defensive decoding guard
        raise MapTilerDownloadError(f"unable to decode tile image: {exc}") from exc

    sanitized_url = str(httpx.URL(str(response.url)).copy_with(query=None))
    return image.convert("RGB"), sanitized_url


def _maptiler_tile_url(*, zoom: int, x: int, y: int) -> str:
    return f"{MAPTILER_TILE_BASE_URL}/{MAPTILER_TILESET}/{zoom}/{x}/{y}.{MAPTILER_IMAGE_EXTENSION}"


def _maptiler_tile_span(start: float, end: float, zoom: int) -> Tuple[float, float, int, int]:
    eps = 1e-9
    scale = float(2 ** zoom)
    start_clamped = min(max(min(start, end), 0.0), scale - eps)
    end_clamped = min(max(max(start, end), 0.0), scale - eps)
    low = int(math.floor(start_clamped))
    high = int(math.floor(max(start_clamped, end_clamped - eps)))
    if high < low:
        high = low
    return start_clamped, end_clamped, low, high


def _maptiler_lon_to_tile_x(lon: float, zoom: int) -> float:
    scale = float(2 ** zoom)
    return (lon + 180.0) / 360.0 * scale


def _maptiler_lat_to_tile_y(lat: float, zoom: int) -> float:
    clamped = _clamp(lat, -MAPTILER_LATITUDE_LIMIT, MAPTILER_LATITUDE_LIMIT)
    sin_lat = math.sin(math.radians(clamped))
    fraction = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
    return fraction * float(2 ** zoom)


def _apply_maptiler_attribution(image: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    text = MAPTILER_ATTRIBUTION_TEXT

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding_x = max(4, base.width // 64)
    padding_y = max(2, base.height // 64)

    box_left = max(0, base.width - text_width - padding_x * 2)
    box_top = max(0, base.height - text_height - padding_y * 2)
    box = (box_left, box_top, base.width, base.height)
    draw.rectangle(box, fill=(0, 0, 0, 160))
    draw.text(
        (box_left + padding_x, box_top + padding_y - bbox[1]),
        text,
        font=font,
        fill=(255, 255, 255, 255),
    )

    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")


def _summarize_maptiler_urls(urls: Sequence[str]) -> str:
    resolved = [url for url in urls if url]
    if not resolved:
        return ""
    if len(resolved) == 1:
        return resolved[0]
    return f"{resolved[0]} (and {len(resolved) - 1} more tiles)"


def _maptiler_fallback_source_summary(
    *,
    zoom: int,
    west: float,
    east: float,
    south: float,
    north: float,
) -> str:
    return (
        f"{MAPTILER_TILE_BASE_URL}/{MAPTILER_TILESET} "
        f"z{zoom} lon[{west:.6f},{east:.6f}] lat[{south:.6f},{north:.6f}]"
    )


def _maptiler_zoom_level(
    *, north: float, south: float, east: float, west: float, width: int, height: int
) -> int:
    lon_span = max(east - west, 1e-9)
    lon_fraction = lon_span / 360.0

    def _mercator_fraction(lat: float) -> float:
        clamped = _clamp(lat, -MAPTILER_LATITUDE_LIMIT, MAPTILER_LATITUDE_LIMIT)
        sin_lat = math.sin(math.radians(clamped))
        return 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)

    north_fraction = _mercator_fraction(north)
    south_fraction = _mercator_fraction(south)
    lat_fraction = max(abs(north_fraction - south_fraction), 1e-9)

    width = max(width, 1)
    height = max(height, 1)

    zoom_x = math.log2(width / (256 * lon_fraction))
    zoom_y = math.log2(height / (256 * lat_fraction))
    zoom = min(zoom_x, zoom_y)
    if not math.isfinite(zoom):
        zoom = MAPTILER_MAX_ZOOM

    zoom = max(0.0, min(float(MAPTILER_MAX_ZOOM), zoom))
    return int(math.floor(zoom))


def _maptiler_native_dim() -> float:
    return max(MIN_DIM, MAPTILER_MIN_TILE_PIXELS / MAPTILER_PIXELS_PER_DEGREE)


def _naip_tile_pixels(span: float) -> int:
    estimated = max(int(round(span * NAIP_PIXELS_PER_DEGREE)), NAIP_MIN_TILE_PIXELS)
    if estimated % 2:
        estimated += 1
    return min(estimated, NAIP_MAX_TILE_PIXELS)


def _naip_tile_intersects_coverage(
    south: float, north: float, west: float, east: float
) -> bool:
    lat_min, lat_max = NAIP_COVERAGE_LAT_RANGE
    lon_min, lon_max = NAIP_COVERAGE_LON_RANGE
    return not (north < lat_min or south > lat_max or east < lon_min or west > lon_max)


def _maptiler_http_error_detail(response: httpx.Response) -> str:
    """Summarize error responses from the MapTiler imagery endpoints."""

    header_detail = response.headers.get("statustext")
    if header_detail:
        header_detail = header_detail.strip()
        if header_detail:
            return _short_error_detail(header_detail)

    content_type = response.headers.get("Content-Type", "").lower()
    content_bytes = getattr(response, "content", b"")
    content_length = len(content_bytes) if content_bytes else 0

    if "image" in content_type:
        if response.status_code in {401, 403}:
            return (
                "forbidden (MapTiler returned an error image). Verify that the MAPTILER_API_KEY "
                "environment variable is set correctly, the key includes Raster/Rendered tile "
                "access, and the attribution/logo overlay remains enabled for free plans."
            )
        return f"{content_type} payload ({content_length} bytes)"

    if "application/json" in content_type or "text/json" in content_type:
        try:
            payload = response.json()
        except Exception:  # pragma: no cover - fallback for unexpected payloads
            return _short_error_detail(response.text)
        else:
            if isinstance(payload, dict):
                for key in ("message", "error", "detail", "description", "error_description"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return _short_error_detail(value)
            return _short_error_detail(str(payload))

    return _short_error_detail(response.text)


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
