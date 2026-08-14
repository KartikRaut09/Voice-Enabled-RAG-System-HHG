"""Health check endpoint."""

from datetime import datetime, timezone

from fastapi import APIRouter

from backend.app.core.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Return application health status."""
    settings = get_settings()
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
