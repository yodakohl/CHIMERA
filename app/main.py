from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, delete, select

from .database import DATA_DIR, get_session, init_db
from .models import AnalysisResult, ApiUsageStat
from .services.analyzer import DEFAULT_PROMPT, get_analyzer, serialize_detections
from .services.imagery import (
    AreaTile,
    GIBS_DEFAULT_LAYER,
    MAX_DIM,
    MIN_DIM,
    MAPTILER_LATITUDE_LIMIT,
    ImageryProviderKey,
    PROVIDER_METADATA,
    download_area_tiles,
    ImageryCancellationError,
)
from pydantic import BaseModel

app = FastAPI(title="Satellite Infrastructure Scanner", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AREA_SCAN_DIR = UPLOAD_DIR / "area_scans"
AREA_SCAN_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SCAN_BOUNDS = {
    "north": 48.2238,
    "south": 48.1938,
    "west": 16.3588,
    "east": 16.3888,
}
# A 0.01° tile size keeps the default Vienna area focused while providing more detail by default.
DEFAULT_SCAN_TILE_SIZE = 0.01
DEFAULT_SCAN_DATE: str | None = None
DEFAULT_IMAGERY_PROVIDER = ImageryProviderKey.MAPTILER_SATELLITE


PROVIDER_OPTIONS = [
    {
        "key": key.value,
        "label": metadata["label"],
        "description": metadata["description"],
    }
    for key, metadata in PROVIDER_METADATA.items()
]

logger = logging.getLogger(__name__)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/data/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class StopScanRequest(BaseModel):
    scan_id: str


class ReclassifyTilePayload(BaseModel):
    image: str
    lat: float | None = None
    lon: float | None = None
    bounds: Dict[str, float] | None = None
    degree_size: float | None = None
    provider_label: str | None = None


class ReclassifyRequest(BaseModel):
    tiles: List[ReclassifyTilePayload]
    prompt: str = DEFAULT_PROMPT


class ScanController:
    def __init__(self) -> None:
        self.cancel_event = asyncio.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


_active_scans: Dict[str, ScanController] = {}
_scan_lock = asyncio.Lock()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(min(value, maximum), minimum)


def _tile_bounds_payload(tile: AreaTile, fallback_dim: float) -> tuple[Dict[str, float], float]:
    try:
        span = float(tile.degree_size)
    except (TypeError, ValueError):
        span = 0.0
    if span <= 0:
        try:
            span = float(fallback_dim)
        except (TypeError, ValueError):
            span = 0.0
    if span <= 0:
        span = DEFAULT_SCAN_TILE_SIZE

    max_latitude = 90.0
    if tile.provider == ImageryProviderKey.MAPTILER_SATELLITE.value:
        max_latitude = MAPTILER_LATITUDE_LIMIT

    half = span / 2.0
    north = _clamp(tile.lat + half, -max_latitude, max_latitude)
    south = _clamp(tile.lat - half, -max_latitude, max_latitude)
    east = _clamp(tile.lon + half, -180.0, 180.0)
    west = _clamp(tile.lon - half, -180.0, 180.0)
    bounds = {"north": north, "south": south, "east": east, "west": west}
    return bounds, span


def _resolve_cached_image(image_reference: str) -> tuple[Path, str]:
    reference = (image_reference or "").strip()
    if not reference:
        raise ValueError("Image path is required")

    prefix = "/data/uploads/"
    if reference.startswith(prefix):
        reference = reference[len(prefix) :]

    relative_path = Path(reference.lstrip("/"))
    if relative_path.is_absolute() or any(part == ".." for part in relative_path.parts):
        raise ValueError("Invalid image path")

    uploads_root = UPLOAD_DIR.resolve()
    candidate = (uploads_root / relative_path).resolve()
    try:
        candidate.relative_to(uploads_root)
    except ValueError as exc:
        raise ValueError("Image path must be within the uploads directory") from exc

    if not candidate.exists():
        raise FileNotFoundError(f"Cached image not found: {relative_path}")

    return candidate, str(relative_path).replace("\\", "/")


async def _register_scan(scan_id: str) -> ScanController:
    async with _scan_lock:
        if scan_id in _active_scans:
            raise HTTPException(status_code=409, detail="Scan already in progress")
        controller = ScanController()
        _active_scans[scan_id] = controller
        return controller


async def _lookup_scan(scan_id: str) -> ScanController | None:
    async with _scan_lock:
        return _active_scans.get(scan_id)


async def _unregister_scan(scan_id: str) -> None:
    async with _scan_lock:
        _active_scans.pop(scan_id, None)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, session: Session = Depends(get_session)):
    return _render_home(request, session)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    session: Session = Depends(get_session),
):
    analyzer = get_analyzer()
    saved_path = await _save_upload(image)
    analysis = analyzer.analyze(saved_path, prompt)

    record = AnalysisResult(
        image_filename=saved_path.name,
        prompt=analysis["prompt"],
        caption=analysis["caption"],
        unusual_summary=analysis["unusual_summary"],
        detection_payload=serialize_detections(analysis["detections"]),
        created_at=datetime.utcnow(),
    )

    session.add(record)
    session.commit()

    return _render_home(request, session, message="Analysis completed")


@app.post("/scan-area", response_class=HTMLResponse)
async def scan_area(
    request: Request,
    north: float = Form(DEFAULT_SCAN_BOUNDS["north"]),
    south: float = Form(DEFAULT_SCAN_BOUNDS["south"]),
    east: float = Form(DEFAULT_SCAN_BOUNDS["east"]),
    west: float = Form(DEFAULT_SCAN_BOUNDS["west"]),
    tile_size: float = Form(DEFAULT_SCAN_TILE_SIZE),
    provider: ImageryProviderKey = Form(DEFAULT_IMAGERY_PROVIDER),
    prompt: str = Form(DEFAULT_PROMPT),
    date: str | None = Form(DEFAULT_SCAN_DATE),
    session: Session = Depends(get_session),
):
    analyzer = get_analyzer()
    area_dir = AREA_SCAN_DIR / datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

    requested_date = (date or "").strip() or None
    provider_meta = PROVIDER_METADATA.get(provider, {})
    provider_label = provider_meta.get("label", provider.value)

    try:
        tiles, download_failures = await download_area_tiles(
            provider=provider,
            north=north,
            south=south,
            east=east,
            west=west,
            dim=tile_size,
            output_dir=area_dir,
            date=requested_date,
        )
    except ValueError as exc:
        if area_dir.exists():
            try:
                area_dir.rmdir()
            except OSError:
                shutil.rmtree(area_dir, ignore_errors=True)
        return _render_home(
            request,
            session,
            message=str(exc),
            selected_provider=provider.value,
        )

    if not tiles:
        if area_dir.exists():
            try:
                next(area_dir.iterdir())
            except StopIteration:
                area_dir.rmdir()
        base_message = "No imagery tiles were downloaded for the requested area."
        if download_failures:
            failures_count = len(download_failures)
            failure_plural = "s" if failures_count != 1 else ""
            base_message += f" {failures_count} imagery request{failure_plural} failed."
            first_failure = download_failures[0]
            base_message += f" First failure detail: {first_failure}."
        if provider_label:
            base_message += f" Provider: {provider_label}."
        return _render_home(
            request,
            session,
            message=base_message,
            selected_provider=provider.value,
        )

    processed_records: List[AnalysisResult] = []
    analysis_failures: List[str] = []

    for tile in tiles:
        try:
            analysis = analyzer.analyze(tile.path, prompt)
        except Exception as exc:  # pragma: no cover - model failure
            logger.exception(
                "Analyzer failed for tile at lat %s lon %s: %s", tile.lat, tile.lon, exc
            )
            analysis_failures.append(f"{tile.lat:.4f}, {tile.lon:.4f}")
            tile.path.unlink(missing_ok=True)
            continue

        record = AnalysisResult(
            image_filename=str(tile.path.relative_to(UPLOAD_DIR)),
            prompt=analysis["prompt"],
            caption=analysis["caption"],
            unusual_summary=analysis["unusual_summary"],
            detection_payload=serialize_detections(analysis["detections"]),
            created_at=datetime.utcnow(),
        )
        processed_records.append(record)

    for record in processed_records:
        session.add(record)
    session.commit()

    processed_count = len(processed_records)
    if processed_count == 0 and area_dir.exists():
        try:
            next(area_dir.iterdir())
        except StopIteration:
            area_dir.rmdir()

    summary = _build_scan_summary_details(
        tiles=tiles,
        tile_size=tile_size,
        provider=provider,
        provider_label=provider_label,
        download_failures=download_failures,
        analysis_failures=analysis_failures,
        processed_count=processed_count,
    )
    message = str(summary.get("message", "Area scan completed."))
    return _render_home(
        request,
        session,
        message=message,
        selected_provider=provider.value,
    )


@app.get("/scan-area/stream")
async def stream_scan_area(
    request: Request,
    scan_id: str = Query(..., description="Unique client-generated identifier for the scan"),
    north: float = Query(DEFAULT_SCAN_BOUNDS["north"]),
    south: float = Query(DEFAULT_SCAN_BOUNDS["south"]),
    east: float = Query(DEFAULT_SCAN_BOUNDS["east"]),
    west: float = Query(DEFAULT_SCAN_BOUNDS["west"]),
    tile_size: float = Query(DEFAULT_SCAN_TILE_SIZE, alias="tile_size"),
    provider: ImageryProviderKey = Query(DEFAULT_IMAGERY_PROVIDER),
    prompt: str = Query(DEFAULT_PROMPT),
    date: str | None = Query(DEFAULT_SCAN_DATE),
    session: Session = Depends(get_session),
):
    scan_id = scan_id.strip()
    if not scan_id:
        raise HTTPException(status_code=400, detail="scan_id query parameter is required")

    controller = await _register_scan(scan_id)
    analyzer = get_analyzer()
    area_dir = AREA_SCAN_DIR / datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

    requested_date = (date or "").strip() or None
    provider_meta = PROVIDER_METADATA.get(provider, {})
    provider_label = provider_meta.get("label", provider.value)

    events: asyncio.Queue[tuple[str, Dict[str, object]]] = asyncio.Queue()
    processed_count = 0
    download_failures: List[str] = []
    analysis_failures: List[str] = []

    async def enqueue(event: str, data: Dict[str, object]) -> None:
        await events.put((event, data))

    async def handle_tile(tile: AreaTile) -> None:
        nonlocal processed_count
        if controller.cancelled:
            raise ImageryCancellationError("Scan cancelled")

        try:
            analysis = analyzer.analyze(tile.path, prompt)
        except Exception as exc:  # pragma: no cover - model failure
            logger.exception(
                "Analyzer failed for tile at lat %s lon %s: %s", tile.lat, tile.lon, exc
            )
            failure_message = (
                f"Analysis failed for tile at {tile.lat:.4f}, {tile.lon:.4f}"
            )
            analysis_failures.append(failure_message)
            bounds, span = _tile_bounds_payload(tile, tile_size)
            await enqueue(
                "analysis-failed",
                {
                    "message": failure_message,
                    "lat": tile.lat,
                    "lon": tile.lon,
                    "bounds": bounds,
                    "degree_size": span,
                },
            )
            tile.path.unlink(missing_ok=True)
            return

        record = AnalysisResult(
            image_filename=str(tile.path.relative_to(UPLOAD_DIR)),
            prompt=analysis["prompt"],
            caption=analysis["caption"],
            unusual_summary=analysis["unusual_summary"],
            detection_payload=serialize_detections(analysis["detections"]),
            created_at=datetime.utcnow(),
        )
        session.add(record)
        session.commit()

        processed_count += 1
        image_relative = str(tile.path.relative_to(UPLOAD_DIR))
        bounds, span = _tile_bounds_payload(tile, tile_size)
        await enqueue(
            "tile",
            {
                "index": processed_count,
                "lat": tile.lat,
                "lon": tile.lon,
                "caption": analysis["caption"],
                "unusual_summary": analysis["unusual_summary"],
                "detections": analysis["detections"],
                "image": f"/data/uploads/{image_relative}",
                "timestamp": record.created_at.isoformat(),
                "provider_label": provider_label,
                "bounds": bounds,
                "degree_size": span,
                "low_detail": tile.low_detail,
                "detail_ratio": tile.detail_ratio,
            },
        )

    async def handle_failure(message: str) -> None:
        if message not in download_failures:
            download_failures.append(message)
        await enqueue("download-failed", {"message": message})

    async def producer() -> None:
        tiles: List[AreaTile] = []
        tile_failures: List[str] = []
        try:
            await enqueue(
                "status",
                {
                    "message": f"Starting scan using {provider_label} imagery.",
                    "provider_label": provider_label,
                },
            )
            tiles, tile_failures = await download_area_tiles(
                provider=provider,
                north=north,
                south=south,
                east=east,
                west=west,
                dim=tile_size,
                output_dir=area_dir,
                date=requested_date,
                progress_callback=handle_tile,
                failure_callback=handle_failure,
                cancel_event=controller.cancel_event,
            )
        except ImageryCancellationError:
            await enqueue("cancelled", {"message": "Scan cancelled."})
        except ValueError as exc:
            await enqueue("error", {"message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Area scan failed: %s", exc)
            await enqueue(
                "error",
                {"message": "Unexpected error while scanning the requested area."},
            )
        else:
            for failure in tile_failures:
                if failure not in download_failures:
                    download_failures.append(failure)
            download_failures_list = list(download_failures)
            if not tiles:
                base_message = "No imagery tiles were downloaded for the requested area."
                if download_failures_list:
                    failures_count = len(download_failures_list)
                    failure_plural = "s" if failures_count != 1 else ""
                    base_message += (
                        f" {failures_count} imagery request{failure_plural} failed."
                    )
                    base_message += f" First failure detail: {download_failures_list[0]}."
                if provider_label:
                    base_message += f" Provider: {provider_label}."
                await enqueue(
                    "complete",
                    {
                        "message": base_message,
                        "provider_label": provider_label,
                        "processed_tiles": processed_count,
                        "download_failures": download_failures_list,
                        "analysis_failures": analysis_failures,
                    },
                )
            else:
                summary = _build_scan_summary_details(
                    tiles=tiles,
                    tile_size=tile_size,
                    provider=provider,
                    provider_label=provider_label,
                    download_failures=download_failures_list,
                    analysis_failures=analysis_failures,
                    processed_count=processed_count,
                )
                await enqueue("complete", summary)
        finally:
            if not processed_count and area_dir.exists():
                try:
                    next(area_dir.iterdir())
                except StopIteration:
                    area_dir.rmdir()
            await enqueue("_end", {})

    async def event_stream() -> AsyncIterator[bytes]:
        producer_task = asyncio.create_task(producer())
        try:
            while True:
                if await request.is_disconnected():
                    controller.cancel()
                try:
                    event_type, payload = await asyncio.wait_for(events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if controller.cancelled and events.empty():
                        break
                    continue
                if event_type == "_end":
                    break
                yield _sse_event(event_type, payload)
        finally:
            controller.cancel()
            await producer_task
            await _unregister_scan(scan_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/scan-area/stop")
async def stop_scan(request: StopScanRequest) -> Dict[str, object]:
    scan_id = request.scan_id.strip()
    if not scan_id:
        raise HTTPException(status_code=400, detail="scan_id is required")

    controller = await _lookup_scan(scan_id)
    if controller is None:
        return {"status": "not_found"}

    controller.cancel()
    return {"status": "stopping"}


@app.post("/scan-area/reclassify")
def reclassify_tiles(request: ReclassifyRequest, session: Session = Depends(get_session)) -> Dict[str, object]:
    if not request.tiles:
        return {"processed": 0, "results": [], "errors": []}

    analyzer = get_analyzer()
    prompt = (request.prompt or "").strip() or DEFAULT_PROMPT

    records: List[AnalysisResult] = []
    results: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    for tile in request.tiles:
        if not tile.image:
            errors.append({"message": "Missing image reference."})
            continue

        try:
            image_path, relative_path = _resolve_cached_image(tile.image)
        except FileNotFoundError as exc:
            errors.append({"image": tile.image, "message": str(exc)})
            continue
        except ValueError as exc:
            errors.append({"image": tile.image, "message": str(exc)})
            continue

        try:
            analysis = analyzer.analyze(image_path, prompt)
        except Exception as exc:  # pragma: no cover - model failure
            logger.exception("Reclassification failed for cached tile %s: %s", image_path, exc)
            errors.append(
                {
                    "image": tile.image,
                    "message": "Analysis failed for cached tile.",
                }
            )
            continue

        created_at = datetime.utcnow()
        record = AnalysisResult(
            image_filename=relative_path,
            prompt=analysis["prompt"],
            caption=analysis["caption"],
            unusual_summary=analysis["unusual_summary"],
            detection_payload=serialize_detections(analysis["detections"]),
            created_at=created_at,
        )
        records.append(record)

        results.append(
            {
                "image": f"/data/uploads/{relative_path}",
                "caption": analysis["caption"],
                "unusual_summary": analysis["unusual_summary"],
                "detections": analysis["detections"],
                "timestamp": created_at.isoformat(),
                "lat": tile.lat,
                "lon": tile.lon,
                "bounds": tile.bounds,
                "degree_size": tile.degree_size,
                "provider_label": tile.provider_label,
            }
        )

    for record in records:
        session.add(record)
    session.commit()

    return {"processed": len(results), "results": results, "errors": errors, "prompt": prompt}


@app.post("/analysis/clear")
def clear_analysis_history(session: Session = Depends(get_session)) -> Dict[str, object]:
    result = session.exec(delete(AnalysisResult))
    session.commit()
    cleared = result.rowcount if result and result.rowcount is not None else 0
    return {"status": "ok", "cleared": cleared}


def _build_scan_summary_details(
    *,
    tiles: List[AreaTile],
    tile_size: float,
    provider: ImageryProviderKey,
    provider_label: str | None,
    download_failures: List[str],
    analysis_failures: List[str],
    processed_count: int,
) -> Dict[str, object]:
    summary_parts: List[str] = []
    if provider_label:
        summary_parts.append(f"Imagery provider: {provider_label}.")
    tile_resolution = tiles[0].pixel_size if tiles else None
    tile_span = tiles[0].degree_size if tiles else None
    tile_layer = tiles[0].layer if tiles else None
    native_tile_span = tiles[0].native_dim if tiles else None
    low_detail_remaining = any(tile.low_detail for tile in tiles)
    base_span = max(tile_size, native_tile_span) if native_tile_span is not None else tile_size
    approx_resolution_m = (
        111_320 / tiles[0].pixels_per_degree if tiles and tiles[0].pixels_per_degree else None
    )
    layer_description = tiles[0].layer_description if tiles else None
    detail_ratios = [tile.detail_ratio for tile in tiles if tile.detail_ratio is not None]
    avg_ratio = None
    if processed_count:
        processed_plural = "s" if processed_count != 1 else ""
        summary_parts.append(f"Analyzed {processed_count} imagery tile{processed_plural}.")
    if tile_layer:
        description = layer_description or tile_layer
        summary_parts.append(f"Imagery layer: {description}.")
        if provider == ImageryProviderKey.NASA_GIBS and tile_layer != GIBS_DEFAULT_LAYER:
            summary_parts.append(
                "Switched to a higher-resolution NASA mosaic after the default VIIRS imagery appeared too blocky."
            )
    if tile_span is not None:
        summary_parts.append(f"Tile span: {tile_span:.3f}°.")
        if abs(tile_span - tile_size) > 1e-6 and native_tile_span is not None:
            if tile_span > base_span + 1e-9:
                summary_parts.append(
                    "Tiles were merged into larger requests after detecting blocky imagery."
                )
            elif tile_span > tile_size:
                summary_parts.append(
                    "Tile span increased from requested "
                    f"{tile_size:.3f}° to {tile_span:.3f}° to match the layer's "
                    f"native resolution (~{native_tile_span:.3f}°)."
                )
    if tile_resolution:
        summary_parts.append(f"Tile resolution: {tile_resolution}px per side.")
    if approx_resolution_m is not None:
        summary_parts.append(
            f"Approximate source resolution: ~{approx_resolution_m:.0f} m per pixel."
        )
    if detail_ratios:
        avg_ratio = sum(detail_ratios) / len(detail_ratios)
        summary_parts.append(
            f"Average neighbor similarity: {avg_ratio:.3f} (lower indicates higher detail)."
        )
    if low_detail_remaining:
        if tile_span is not None and tile_span >= MAX_DIM - 1e-9:
            summary_parts.append(
                "Imagery still appears low detail even at the maximum supported tile span; "
                "higher resolution data may not be available from the source."
            )
        else:
            summary_parts.append(
                "Some tiles still appear low detail; consider enlarging the coverage further."
            )
    download_failures_count = len(download_failures)
    if download_failures_count:
        download_plural = "s" if download_failures_count != 1 else ""
        summary_parts.append(f"{download_failures_count} download{download_plural} failed.")
        summary_parts.append(f"Example failure: {download_failures[0]}.")
    analysis_failures_count = len(analysis_failures)
    if analysis_failures_count:
        analysis_plural = "s" if analysis_failures_count != 1 else ""
        summary_parts.append(
            f"Failed to analyze {analysis_failures_count} tile{analysis_plural}."
        )

    message = " ".join(summary_parts) or "Area scan completed."
    return {
        "message": message,
        "provider_label": provider_label,
        "processed_tiles": processed_count,
        "tile_span": tile_span,
        "tile_resolution": tile_resolution,
        "approx_resolution_m": approx_resolution_m,
        "tile_layer": tile_layer,
        "layer_description": layer_description,
        "download_failures": download_failures,
        "analysis_failures": analysis_failures,
        "low_detail_remaining": low_detail_remaining,
        "average_detail_ratio": avg_ratio,
    }


def _sse_event(event: str, data: Dict[str, object]) -> bytes:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".png"
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in {"-", "_"}) or "upload"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{safe_stem}_{timestamp}{suffix}"


_NO_UNUSUAL_FINDINGS_MARKERS = {
    "nothing unusual",
    "no unusual",
    "nothing out of the ordinary",
    "nothing out of ordinary",
    "no anomalies",
    "no unusual objects",
    "no unusual activity",
    "nothing noteworthy",
    "nothing of note",
    "none",
}
_UNUSUAL_SEGMENT_PATTERN = re.compile(r"(?:,|;|\\.|\\band\\b)", flags=re.IGNORECASE)


def _unusual_summary_score(summary: str | None) -> int:
    if not summary:
        return 0

    normalized = summary.strip().lower()
    if not normalized:
        return 0

    if any(marker in normalized for marker in _NO_UNUSUAL_FINDINGS_MARKERS):
        return 0

    segments = [
        segment.strip()
        for segment in _UNUSUAL_SEGMENT_PATTERN.split(summary)
        if segment.strip()
    ]
    return len(segments) or 1


def _analysis_sort_key(
    result: AnalysisResult, detections: List[Dict[str, object]]
) -> tuple[int, int, datetime]:
    summary_score = _unusual_summary_score(result.unusual_summary)
    detection_score = len(detections)
    timestamp = result.created_at or datetime.min
    return (summary_score, detection_score, timestamp)


def _build_results(session: Session) -> List[Dict[str, object]]:
    statement = select(AnalysisResult)
    results: List[AnalysisResult] = session.exec(statement).all()

    decorated: List[tuple[tuple[int, int, datetime], AnalysisResult, List[Dict[str, object]]]] = []
    for result in results:
        detections = result.detections()
        sort_key = _analysis_sort_key(result, detections)
        decorated.append((sort_key, result, detections))

    decorated.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "id": result.id,
            "image_filename": result.image_filename,
            "prompt": result.prompt,
            "caption": result.caption,
            "unusual_summary": result.unusual_summary,
            "detections": detections,
            "created_at": result.created_at,
        }
        for _, result, detections in decorated
    ]


def _provider_label_for(provider_key: str) -> str:
    for key, metadata in PROVIDER_METADATA.items():
        if key.value == provider_key:
            return metadata.get("label", provider_key)
    return provider_key


def _build_api_usage(session: Session) -> List[Dict[str, object]]:
    statement = select(ApiUsageStat).order_by(ApiUsageStat.provider)
    stats: List[ApiUsageStat] = session.exec(statement).all()
    usage: List[Dict[str, object]] = []
    for stat in stats:
        usage.append(
            {
                "provider": stat.provider,
                "provider_label": _provider_label_for(stat.provider),
                "request_count": stat.request_count,
                "last_used_at": stat.last_used_at,
            }
        )
    return usage


def _render_home(
    request: Request,
    session: Session,
    message: str | None = None,
    *,
    selected_provider: str | None = None,
):
    context: Dict[str, object] = {
        "request": request,
        "default_prompt": DEFAULT_PROMPT,
        "results": _build_results(session),
        "default_bounds": DEFAULT_SCAN_BOUNDS,
        "default_tile_size": DEFAULT_SCAN_TILE_SIZE,
        "default_date": DEFAULT_SCAN_DATE,
        "providers": PROVIDER_OPTIONS,
        "default_provider": DEFAULT_IMAGERY_PROVIDER.value,
        "selected_provider": selected_provider or DEFAULT_IMAGERY_PROVIDER.value,
        "min_tile_size": MIN_DIM,
        "max_tile_size": MAX_DIM,
        "api_usage": _build_api_usage(session),
    }
    if message:
        context["message"] = message
    return templates.TemplateResponse("index.html", context)


async def _save_upload(upload: UploadFile, *, target_dir: Path | None = None) -> Path:
    destination = target_dir or UPLOAD_DIR
    destination.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(upload.filename or "upload.png")
    saved_path = destination / filename
    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return saved_path
