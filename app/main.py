from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select

from .database import DATA_DIR, get_session, init_db
from .models import AnalysisResult
from .services.analyzer import DEFAULT_PROMPT, get_analyzer, serialize_detections
from .services.imagery import (
    GIBS_DEFAULT_LAYER,
    MAX_DIM,
    MIN_DIM,
    download_gibs_area_tiles,
)

app = FastAPI(title="Satellite Infrastructure Scanner", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AREA_SCAN_DIR = UPLOAD_DIR / "area_scans"
AREA_SCAN_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SCAN_BOUNDS = {
    "north": 37.83,
    "south": 37.73,
    "west": -122.52,
    "east": -122.47,
}
# A 0.05° tile size yields a compact 2x1 grid for the default bounding box.
DEFAULT_SCAN_TILE_SIZE = 0.05
DEFAULT_SCAN_DATE: str | None = None

logger = logging.getLogger(__name__)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/data/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


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
    prompt: str = Form(DEFAULT_PROMPT),
    date: str | None = Form(DEFAULT_SCAN_DATE),
    session: Session = Depends(get_session),
):
    analyzer = get_analyzer()
    area_dir = AREA_SCAN_DIR / datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

    requested_date = (date or "").strip() or None

    try:
        tiles, download_failures = await download_gibs_area_tiles(
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
        return _render_home(request, session, message=str(exc))

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
            base_message += f" {failures_count} GIBS request{failure_plural} failed."
            first_failure = download_failures[0]
            base_message += f" First failure detail: {first_failure}."
        return _render_home(request, session, message=base_message)

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

    summary_parts: List[str] = []
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
    if processed_count:
        processed_plural = "s" if processed_count != 1 else ""
        summary_parts.append(f"Analyzed {processed_count} GIBS tile{processed_plural}.")
    if tile_layer:
        description = layer_description or tile_layer
        summary_parts.append(f"Imagery layer: {description}.")
        if tile_layer != GIBS_DEFAULT_LAYER:
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
    return _render_home(request, session, message=message)


def _safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".png"
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in {"-", "_"}) or "upload"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{safe_stem}_{timestamp}{suffix}"


def _build_results(session: Session) -> List[Dict[str, object]]:
    statement = select(AnalysisResult).order_by(AnalysisResult.created_at.desc())
    results: List[AnalysisResult] = session.exec(statement).all()
    return [
        {
            "id": result.id,
            "image_filename": result.image_filename,
            "prompt": result.prompt,
            "caption": result.caption,
            "unusual_summary": result.unusual_summary,
            "detections": result.detections(),
            "created_at": result.created_at,
        }
        for result in results
    ]


def _render_home(request: Request, session: Session, message: str | None = None):
    context: Dict[str, object] = {
        "request": request,
        "default_prompt": DEFAULT_PROMPT,
        "results": _build_results(session),
        "default_bounds": DEFAULT_SCAN_BOUNDS,
        "default_tile_size": DEFAULT_SCAN_TILE_SIZE,
        "default_date": DEFAULT_SCAN_DATE,
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
