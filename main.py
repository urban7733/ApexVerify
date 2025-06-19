from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pathlib import Path
import uvicorn
import os
from datetime import datetime
import uuid
from fastapi.middleware.cors import CORSMiddleware
from services.deepfake_detector import DeepfakeDetector
from services.storage_service import StorageService
from services.report_generator import ReportGenerator
from services.learning_service import LearningService
from services.trust_badge import TrustBadgeService
from services.reinforcement_learning import ReinforcementLearningService
from services.openevolve_integration import OpenEvolveDeepfakeDetector
import shutil
import logging
from typing import Optional
from services.api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
REPORTS_DIR = Path("reports")
STATIC_DIR = Path("static")
BADGE_DIR = Path("static/badges")
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
EVOLUTION_DIR = Path("evolution_output")

for directory in [UPLOAD_DIR, REPORTS_DIR, STATIC_DIR, BADGE_DIR, DATA_DIR, MODELS_DIR, EVOLUTION_DIR]:
    directory.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
deepfake_detector = DeepfakeDetector()
storage_service = StorageService()
report_generator = ReportGenerator()
learning_service = LearningService()
trust_badge_service = TrustBadgeService()
rl_service = ReinforcementLearningService()
openevolve_detector = OpenEvolveDeepfakeDetector(evolution_dir=str(EVOLUTION_DIR))

# Set up reinforcement learning integration
try:
    # Set feature extractor for RL service
    if hasattr(deepfake_detector, 'extract_features'):
        rl_service.set_feature_extractor(deepfake_detector.extract_features)
    logger.info("Reinforcement learning service initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize RL service: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading index.html: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a media file for analysis
    """
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Store file metadata
        metadata = {
            "file_id": file_id,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "upload_time": datetime.utcnow(),
            "file_type": file.content_type
        }
        storage_service.store_metadata(metadata)
        
        return {"file_id": file_id, "message": "File uploaded successfully"}
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    user_feedback: Optional[bool] = None,
    generate_badge: bool = False,
    badge_style: str = "standard",
    use_evolution: bool = True
):
    """
    Analyze media file for deepfake detection with OpenEvolve self-learning
    """
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract features for analysis
        features = None
        if hasattr(deepfake_detector, 'extract_features'):
            try:
                features = deepfake_detector.extract_features(str(file_path))
            except Exception as e:
                logger.warning(f"Could not extract features: {str(e)}")
        
        # Analyze file using OpenEvolve if enabled
        if use_evolution and features is not None:
            # Use OpenEvolve enhanced detection
            metadata = {
                "file_id": file_id,
                "original_filename": file.filename,
                "file_type": file.content_type,
                "media_type": "image" if file.content_type.startswith("image/") else "video",
                "file_size": os.path.getsize(file_path)
            }
            
            # Get ground truth from user feedback if available
            ground_truth = user_feedback if user_feedback is not None else None
            
            # Analyze with evolution
            evolution_result = openevolve_detector.analyze_with_evolution(
                features=features,
                metadata=metadata,
                ground_truth=ground_truth
            )
            
            # Convert to standard format
            result = {
                "is_fake": evolution_result.is_fake,
                "confidence": evolution_result.confidence,
                "prediction": evolution_result.prediction,
                "faces_detected": evolution_result.faces_detected,
                "media_type": evolution_result.media_type,
                "learning_metrics": evolution_result.learning_metrics,
                "evolution_data": evolution_result.evolution_data,
                "gpu_accelerated": torch.cuda.is_available() if 'torch' in globals() else False
            }
            
        else:
            # Use standard analysis
            if file.content_type.startswith("image/"):
                result = deepfake_detector.analyze(str(file_path))
            elif file.content_type.startswith("video/"):
                result = await deepfake_detector.analyze_video(str(file_path))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Process user feedback with reinforcement learning
        if user_feedback is not None and features is not None:
            rl_result = rl_service.process_user_feedback(
                file_id=file_id,
                features=features,
                prediction=result,
                user_feedback=user_feedback
            )
            result["reinforcement_learning"] = rl_result
            
            # Add feedback to OpenEvolve detector
            openevolve_detector.add_feedback(file_id, user_feedback)
        
        # Generate trust badge if requested
        if generate_badge and file.content_type.startswith("image/"):
            badge_path = trust_badge_service.generate_badge(
                result,
                str(BADGE_DIR / f"badge_{file_id}.png"),
                badge_style
            )
            
            if badge_path:
                # Apply badge to image
                watermarked_path = trust_badge_service.apply_badge_to_image(
                    str(file_path),
                    badge_path,
                    str(UPLOAD_DIR / f"watermarked_{file_id}_{file.filename}")
                )
                
                if watermarked_path:
                    result["watermarked_image"] = watermarked_path
                    result["badge_path"] = badge_path
        
        # Store analysis result
        storage_service.store_analysis(file_id, result)
        
        # Clean up original file
        os.remove(file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(
    file_id: str,
    user_feedback: bool,
    confidence_rating: Optional[float] = None
):
    """
    Submit user feedback for reinforcement learning and OpenEvolve
    """
    try:
        # Get stored analysis
        analysis = storage_service.get_analysis(file_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Extract features if available
        features = None
        if hasattr(deepfake_detector, 'extract_features'):
            try:
                metadata = storage_service.get_metadata(file_id)
                if metadata and metadata.get("file_path"):
                    features = deepfake_detector.extract_features(metadata["file_path"])
            except Exception as e:
                logger.warning(f"Could not extract features for feedback: {str(e)}")
        
        # Process feedback with reinforcement learning
        if features is not None:
            rl_result = rl_service.process_user_feedback(
                file_id=file_id,
                features=features,
                prediction=analysis,
                user_feedback=user_feedback
            )
        else:
            rl_result = {"message": "Features not available for learning"}
        
        # Add feedback to OpenEvolve detector
        openevolve_detector.add_feedback(file_id, user_feedback, confidence_rating)
        
        # Store feedback
        feedback_data = {
            "file_id": file_id,
            "user_feedback": user_feedback,
            "confidence_rating": confidence_rating,
            "timestamp": datetime.utcnow().isoformat(),
            "rl_result": rl_result
        }
        storage_service.store_feedback(file_id, feedback_data)
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "rl_result": rl_result
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evolution/status")
async def get_evolution_status():
    """Get OpenEvolve evolution status and statistics"""
    try:
        return openevolve_detector.get_evolution_status()
    except Exception as e:
        logger.error(f"Error getting evolution status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evolution/trigger")
async def trigger_evolution(min_data_points: int = 50):
    """Manually trigger OpenEvolve evolution"""
    try:
        success = openevolve_detector.trigger_evolution(min_data_points)
        return {
            "status": "success" if success else "insufficient_data",
            "triggered": success,
            "data_points_available": len(openevolve_detector.analysis_history)
        }
    except Exception as e:
        logger.error(f"Error triggering evolution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evolution/export")
async def export_evolution_data():
    """Export OpenEvolve learning data"""
    try:
        return openevolve_detector.export_learning_data()
    except Exception as e:
        logger.error(f"Error exporting evolution data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{file_id}")
async def get_report(file_id: str, format: str = "json"):
    """
    Get analysis report in JSON or PDF format
    """
    try:
        analysis_data = storage_service.get_analysis(file_id)
        if not analysis_data:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        if format.lower() == "pdf":
            pdf_path = report_generator.generate_pdf(file_id, analysis_data)
            return FileResponse(pdf_path, media_type="application/pdf")
        
        return analysis_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/insights")
async def get_learning_insights():
    """Get reinforcement learning insights"""
    try:
        return rl_service.get_learning_insights()
    except Exception as e:
        logger.error(f"Error getting learning insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning/enable")
async def enable_learning(enabled: bool = True):
    """Enable or disable reinforcement learning"""
    try:
        rl_service.enable_learning(enabled)
        return {"status": "success", "learning_enabled": enabled}
    except Exception as e:
        logger.error(f"Error enabling learning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning/reset")
async def reset_learning():
    """Reset the reinforcement learning agent"""
    try:
        rl_service.reset_learning()
        return {"status": "success", "message": "Learning agent reset"}
    except Exception as e:
        logger.error(f"Error resetting learning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/export")
async def export_learning_data():
    """Export learning data for analysis"""
    try:
        return rl_service.export_learning_data()
    except Exception as e:
        logger.error(f"Error exporting learning data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reverse-search")
async def reverse_search(file_id: str):
    """
    Perform reverse image search using EagleEye
    """
    try:
        metadata = storage_service.get_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        # TODO: Implement EagleEye integration
        return {"message": "Reverse search not implemented yet"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/badge/{file_id}")
async def get_badged_image(file_id: str):
    """
    Get the media file with trust badge overlay
    """
    try:
        metadata = storage_service.get_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        # TODO: Implement badge overlay
        return {"message": "Badge overlay not implemented yet"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download analyzed file with trust badge"""
    try:
        file_path = UPLOAD_DIR / f"watermarked_{file_id}"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            str(file_path),
            media_type="image/jpeg",
            filename=f"verified_{file_id}.jpg"
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/metrics")
async def get_learning_metrics():
    """Get learning metrics and statistics"""
    try:
        return learning_service.get_learning_metrics()
    except Exception as e:
        logger.error(f"Error getting learning metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/history")
async def get_analysis_history():
    """Get analysis history"""
    try:
        return learning_service.get_analysis_history()
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/thresholds")
async def get_confidence_thresholds():
    """Get current confidence thresholds"""
    try:
        return learning_service.get_confidence_thresholds()
    except Exception as e:
        logger.error(f"Error getting confidence thresholds: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        logger.info("Starting ApexVerify AI server with OpenEvolve integration...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise 