from __future__ import annotations

import logging
import os
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
from .services.imagery import download_nasa_area_tiles

app = FastAPI(title="Satellite Infrastructure Scanner", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AREA_SCAN_DIR = UPLOAD_DIR / "area_scans"
AREA_SCAN_DIR.mkdir(parents=True, exist_ok=True)

NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

DEFAULT_SCAN_BOUNDS = {
    "north": 37.83,
    "south": 37.73,
    "west": -122.52,
    "east": -122.47,
}
# A 0.05° tile size keeps the DEMO_KEY requests at two tiles for the default bounding box.
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
    api_key: str | None = Form(None),
    session: Session = Depends(get_session),
):
    analyzer = get_analyzer()
    area_dir = AREA_SCAN_DIR / datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

    selected_api_key = (api_key or "").strip() or NASA_API_KEY
    requested_date = (date or "").strip() or None

    try:
        tiles, download_failures = await download_nasa_area_tiles(
            north=north,
            south=south,
            east=east,
            west=west,
            dim=tile_size,
            output_dir=area_dir,
            api_key=selected_api_key,
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
            base_message += f" {failures_count} NASA request{failure_plural} failed."
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
    if processed_count:
        processed_plural = "s" if processed_count != 1 else ""
        summary_parts.append(f"Analyzed {processed_count} NASA tile{processed_plural}.")
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
