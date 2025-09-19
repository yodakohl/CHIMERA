from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select

from .database import DATA_DIR, get_session, init_db
from .models import AnalysisResult
from .services.analyzer import DEFAULT_PROMPT, get_analyzer, serialize_detections

app = FastAPI(title="Satellite Infrastructure Scanner", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/data/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, session: Session = Depends(get_session)):
    statement = select(AnalysisResult).order_by(AnalysisResult.created_at.desc())
    results: List[AnalysisResult] = session.exec(statement).all()
    context = {
        "request": request,
        "default_prompt": DEFAULT_PROMPT,
        "results": [
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
        ],
    }
    return templates.TemplateResponse("index.html", context)


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

    statement = select(AnalysisResult).order_by(AnalysisResult.created_at.desc())
    results = session.exec(statement).all()

    context = {
        "request": request,
        "default_prompt": DEFAULT_PROMPT,
        "results": [
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
        ],
        "message": "Analysis completed",
    }
    return templates.TemplateResponse("index.html", context)


def _safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".png"
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in {"-", "_"}) or "upload"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{safe_stem}_{timestamp}{suffix}"


async def _save_upload(upload: UploadFile) -> Path:
    filename = _safe_filename(upload.filename or "upload.png")
    saved_path = UPLOAD_DIR / filename
    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return saved_path
