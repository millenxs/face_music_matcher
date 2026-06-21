"""FastAPI application factory.

Serves the REST API, the frontend SPA, and generated plot images.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.compare import router as compare_router
from app.core.config import config


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance ready to serve.
    """
    app = FastAPI(
        title="Face Music Matcher",
        description=(
            "Experimental computer vision and signal processing system that "
            "compares facial geometry with musical structure using pure "
            "mathematical curve similarity."
        ),
        version="0.2.0",
    )

    # CORS — allow frontend to call the API from any origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve generated plots as static files.
    config.ensure_directories()
    outputs_abs = str(config.OUTPUT_DIR.resolve())
    app.mount("/outputs", StaticFiles(directory=outputs_abs), name="outputs")

    # API routes.
    app.include_router(compare_router)

    # Frontend page.
    @app.get("/", response_class=HTMLResponse)
    async def frontend():
        """Serve the interactive frontend."""
        frontend_path = Path(__file__).resolve().parent / "frontend.html"
        if frontend_path.exists():
            return frontend_path.read_text(encoding="utf-8")
        return "<h1>Frontend not found. Run the API and open /docs for Swagger UI.</h1>"

    return app


app = create_app()
