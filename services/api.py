from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import Dict, Any
import logging
from .deepfake_detector import DeepfakeDetector
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ApexVerify AI API",
    description="API for deepfake detection and reverse image search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = DeepfakeDetector()

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/badges", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading main page")

@app.post("/api/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Analyze uploaded media file for deepfake detection and perform reverse image search
    """
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{timestamp}{file_extension}"
        file_path = os.path.join("uploads", unique_filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Analyze file
        analysis_results = await detector.analyze(file_path)

        # Add cleanup task
        if background_tasks:
            background_tasks.add_task(os.remove, file_path)

        return analysis_results

    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": detector.device.type == "cuda",
        "models_loaded": True
    }

@app.get("/api/models")
async def get_models() -> Dict[str, Any]:
    """Get information about loaded models"""
    return {
        "deepfake_model": detector.deepfake_model_name,
        "face_model": detector.face_model_name,
        "device": str(detector.device)
    }

@app.post("/api/batch-analyze")
async def batch_analyze(
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Analyze multiple files in batch
    """
    try:
        results = []
        for file in files:
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{timestamp}{file_extension}"
            file_path = os.path.join("uploads", unique_filename)

            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Analyze file
            analysis_result = await detector.analyze(file_path)
            results.append({
                "filename": file.filename,
                "analysis": analysis_result
            })

            # Add cleanup task
            if background_tasks:
                background_tasks.add_task(os.remove, file_path)

        return {
            "total_files": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    ) 