from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import select

from ..database import init_db, session_scope
from ..models import ApiUsageStat


_usage_initialized = False


def _ensure_usage_table() -> None:
    global _usage_initialized
    if not _usage_initialized:
        init_db()
        _usage_initialized = True


def record_api_usage(provider: str, *, increment: int = 1) -> None:
    """Increment the API usage counter for the given provider."""

    if increment <= 0:
        return

    _ensure_usage_table()

    with session_scope() as session:
        statement = select(ApiUsageStat).where(ApiUsageStat.provider == provider)
        usage = session.exec(statement).one_or_none()
        now = datetime.now(UTC)
        if usage is None:
            usage = ApiUsageStat(provider=provider, request_count=increment, last_used_at=now)
            session.add(usage)
        else:
            usage.request_count += increment
            usage.last_used_at = now
        session.commit()
