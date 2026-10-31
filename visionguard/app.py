from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse

from visionguard.api.routes import router

app = FastAPI(
    title="VisionGuard Backend API",
    description=(
        "Hardware/PCB optical inspection and circuit extraction engine. "
        "Transforms PCB images and live camera feeds into editable, validated circuits "
        "with SVG/PNG and Wokwi exports."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for pair programming and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_vision_mode_header(request: Request, call_next):
    from visionguard.config import settings
    response = await call_next(request)
    response.headers["X-Vision-Mode"] = settings.VISION_BACKEND
    return response

# Include routes both at root and with /api/v1 prefix
app.include_router(router)
app.include_router(router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def root(request: Request):
    """
    Root endpoint.
    If accessed via a web browser (Accept: text/html), automatically redirects to /docs (Swagger UI).
    Otherwise returns API metadata JSON.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return RedirectResponse(url="/docs")
    return {
        "service": "VisionGuard Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "schema": "/circuits/schema",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("visionguard.app:app", host="0.0.0.0", port=8000, reload=True)

