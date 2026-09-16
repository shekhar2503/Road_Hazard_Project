import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database import check_db_status
from backend.routes.hazards import router as hazards_router
from backend.routes.routing import router as routing_router
from backend.routes.auth import router as auth_router
from starlette.middleware.sessions import SessionMiddleware
from backend.schemas.detection import DetectionResponse, HealthResponse
from backend.services.detector import RoadHazardDetector

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Allowed image MIME types
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager.
    Loads the YOLO model checkpoint once into memory on application startup.
    """
    print("\n=======================================================")
    print("STARTING ROAD HAZARDS DETECTION FASTAPI BACKEND")
    print("=======================================================")
    try:
        app.state.detector = RoadHazardDetector()
        print("[Lifespan] RoadHazardDetector initialized and ready.")
    except Exception as e:
        print(f"[Lifespan Error] Failed to initialize detector: {e}")
        raise e

    # Check database status on startup
    db_status = check_db_status()
    print(f"[Lifespan] PostgreSQL Database Status: {db_status.upper()}")

    yield

    print("[Lifespan] Shutting down Road Hazards Detection Backend.")


# Initialize FastAPI application
app = FastAPI(
    title="Road Hazard Intelligence API",
    description=(
        "Production AI backend for automated road hazard detection "
        "(Potholes, Road Cracks, Waterlogging, Construction Barriers) "
        "powered by fine-tuned YOLO11s at 800x800 resolution with "
        "PostgreSQL + PostGIS spatial persistence."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS for local development frontends (React / Vite / Next.js)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate SESSION_SECRET_KEY
SESSION_SECRET_KEY = os.environ.get("SESSION_SECRET_KEY")
if not SESSION_SECRET_KEY or len(SESSION_SECRET_KEY) < 32:
    raise RuntimeError("SESSION_SECRET_KEY environment variable is not set or is shorter than 32 characters. A long random string is required for session security.")

SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    session_cookie="road_hazard_admin_session",
    max_age=8 * 3600,
    same_site="lax",
    https_only=SESSION_COOKIE_SECURE,
)

# Mount outputs directory for static serving of annotated images
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Register Hazard management routes (CRUD + spatial querying)
app.include_router(hazards_router, prefix="/hazards")

# Register Hazard-aware routing recommendation routes
app.include_router(routing_router, prefix="/route")

# Register Authentication routes
app.include_router(auth_router, prefix="/api")


@app.get("/", tags=["General"])
async def root():
    """Root landing endpoint with system info and links to documentation."""
    return {
        "service": "Road Hazard Intelligence System API",
        "version": "1.1.0",
        "documentation": "/docs",
        "health_check": "/health",
        "inference_endpoint": "/detect/image",
        "hazards_endpoint": "/hazards",
        "routing_endpoint": "/route/recommend",
        "database_status": check_db_status(),
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint. Returns operational status, model info, device, and database status."""
    detector: RoadHazardDetector = getattr(app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model detector service is not initialized or still starting up.",
        )

    health_info = detector.get_health()
    # Check database status dynamically
    health_info.database = check_db_status()
    return health_info


@app.post(
    "/detect/image",
    response_model=DetectionResponse,
    tags=["Inference"],
    summary="Detect road hazards in an uploaded image",
)
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, WEBP, BMP)"),
    conf: float = Query(
        0.25,
        ge=0.01,
        le=1.0,
        description="Confidence threshold for bounding box predictions",
    ),
    iou: float = Query(
        0.45,
        ge=0.01,
        le=1.0,
        description="Non-Maximum Suppression (NMS) IoU threshold",
    ),
):
    """Analyze an uploaded road image and detect hazards.
    Returns bounding boxes, class names, confidence scores, and an annotated image URL.
    (Note: Inference is decoupled from hazard persistence; save hazards via POST /hazards).
    """
    detector: RoadHazardDetector = getattr(app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detector model is not available.",
        )

    # 1. Validate file content-type and extension
    filename = file.filename or "upload.jpg"
    ext = os.path.splitext(filename)[1].lower()

    is_valid_type = (
        file.content_type in ALLOWED_IMAGE_TYPES
        or (file.content_type and file.content_type.startswith("image/"))
        or ext in ALLOWED_EXTENSIONS
    )

    if not is_valid_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file format '{file.content_type}'. "
                f"Please upload a valid image file ({', '.join(sorted(ALLOWED_EXTENSIONS))})."
            ),
        )

    # 2. Read file bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 3. Execute inference
    try:
        result = detector.predict_image(
            image_bytes=image_bytes,
            filename=filename,
            conf_threshold=conf,
            iou_threshold=iou,
            output_dir=OUTPUTS_DIR,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid image content: {str(val_err)}",
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(err)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
