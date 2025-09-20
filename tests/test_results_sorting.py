import json
from datetime import datetime, timedelta

from sqlmodel import Session, SQLModel, create_engine

from app.main import _build_results
from app.models import AnalysisResult


def create_memory_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def test_build_results_sorted_by_recency():
    engine = create_memory_session()
    with Session(engine) as session:
        base = datetime(2024, 1, 1, 12, 0, 0)
        for index in range(3):
            session.add(
                AnalysisResult(
                    image_filename=f"image_{index}.jpg",
                    prompt="prompt",
                    caption=f"Caption {index}",
                    unusual_summary=f"Summary {index}",
                    detection_payload="[]",
                    created_at=base + timedelta(minutes=index),
                )
            )
        session.commit()

        results = _build_results(session)
        timestamps = [entry["created_at"] for entry in results]
        assert timestamps == sorted(timestamps, reverse=True)


def test_interesting_entries_win_ties_on_same_timestamp():
    engine = create_memory_session()
    with Session(engine) as session:
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        boring = AnalysisResult(
            image_filename="boring.jpg",
            prompt="prompt",
            caption="Boring",
            unusual_summary="Nothing unusual to report.",
            detection_payload="[]",
            created_at=timestamp,
        )
        interesting = AnalysisResult(
            image_filename="interesting.jpg",
            prompt="prompt",
            caption="Interesting",
            unusual_summary="Possible new structure detected near the runway.",
            detection_payload=json.dumps(
                [{"object": "tower", "confidence": 0.91}]
            ),
            created_at=timestamp,
        )
        session.add(boring)
        session.add(interesting)
        session.commit()

        results = _build_results(session)
        captions = [entry["caption"] for entry in results]
        assert captions[0] == "Interesting"
        assert captions[1] == "Boring"
