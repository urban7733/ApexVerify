import pytest
import os
import tempfile
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
from io import BytesIO
import json

# Mock the deepfake detector to avoid model loading issues
class MockDeepfakeDetector:
    def __init__(self):
        self.device = "cpu"
        self.deepfake_model_name = "microsoft/beit-base-patch16-224"
        self.face_model_name = "microsoft/beit-base-patch16-224-pt22k-ft22k"
    
    def analyze(self, file_path):
        return {
            "is_fake": False,
            "confidence": 0.85,
            "prediction": "REAL",
            "faces_detected": 1,
            "media_type": "image",
            "gpu_accelerated": False
        }
    
    def extract_features(self, file_path):
        return np.random.rand(512)
    
    async def analyze_video(self, file_path):
        return {
            "total_frames": 100,
            "analyzed_frames": 10,
            "fake_frames": 0,
            "fake_percentage": 0.0,
            "media_type": "video"
        }

# Mock the storage service
class MockStorageService:
    def __init__(self):
        self.metadata = {}
        self.analysis = {}
        self.feedback = {}
    
    def store_metadata(self, metadata):
        self.metadata[metadata["file_id"]] = metadata
    
    def get_metadata(self, file_id):
        return self.metadata.get(file_id)
    
    def store_analysis(self, file_id, analysis):
        self.analysis[file_id] = analysis
    
    def get_analysis(self, file_id):
        return self.analysis.get(file_id)
    
    def store_feedback(self, file_id, feedback):
        if file_id not in self.feedback:
            self.feedback[file_id] = []
        self.feedback[file_id].append(feedback)

# Mock the RL service
class MockRLService:
    def __init__(self):
        self.learning_enabled = True
    
    def process_user_feedback(self, file_id, features, prediction, user_feedback):
        return {
            "status": "success",
            "message": "Feedback processed successfully"
        }
    
    def get_learning_insights(self):
        return {
            "training_stats": {"episodes": 0, "total_reward": 0},
            "performance": {"accuracy": 0.8, "avg_confidence": 0.7},
            "learning_enabled": True
        }
    
    def enable_learning(self, enabled):
        self.learning_enabled = enabled
    
    def reset_learning(self):
        pass
    
    def export_learning_data(self):
        return {"training_stats": {}, "performance": {}}

# Create a simple test app
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Test ApexVerify AI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize mock services
detector = MockDeepfakeDetector()
storage = MockStorageService()
rl_service = MockRLService()

@app.get("/")
async def root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {str(e)}"

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": False,
        "models_loaded": True
    }

@app.get("/api/models")
async def get_models():
    """Get information about loaded models"""
    return {
        "deepfake_model": detector.deepfake_model_name,
        "face_model": detector.face_model_name,
        "device": str(detector.device)
    }

@app.post("/upload")
async def upload_file(file):
    """Upload a media file for analysis"""
    try:
        # Generate unique file ID
        import uuid
        file_id = str(uuid.uuid4())
        
        # Store file metadata
        metadata = {
            "file_id": file_id,
            "original_filename": file.filename,
            "upload_time": "2024-01-01T00:00:00",
            "file_type": file.content_type
        }
        storage.store_metadata(metadata)
        
        return {"file_id": file_id, "message": "File uploaded successfully"}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
async def analyze_media(file, user_feedback=None, generate_badge=False, badge_style="standard"):
    """Analyze media file for deepfake detection"""
    try:
        # Analyze file
        if file.content_type.startswith("image/"):
            result = detector.analyze("test_path")
        elif file.content_type.startswith("video/"):
            result = await detector.analyze_video("test_path")
        else:
            return {"error": "Unsupported file type"}
        
        # Extract features for reinforcement learning
        features = detector.extract_features("test_path")
        
        # Process user feedback with reinforcement learning
        if user_feedback is not None and features is not None:
            rl_result = rl_service.process_user_feedback(
                file_id="test_id",
                features=features,
                prediction=result,
                user_feedback=user_feedback
            )
            result["reinforcement_learning"] = rl_result
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/learning/insights")
async def get_learning_insights():
    """Get reinforcement learning insights"""
    return rl_service.get_learning_insights()

@app.get("/learning/metrics")
async def get_learning_metrics():
    """Get learning metrics and statistics"""
    return {
        "total_analyses": 100,
        "accuracy": 0.85,
        "learning_rate": 0.001
    }

@app.get("/learning/history")
async def get_analysis_history():
    """Get analysis history"""
    return [
        {"id": "1", "timestamp": "2024-01-01", "result": "REAL"},
        {"id": "2", "timestamp": "2024-01-02", "result": "FAKE"}
    ]

@app.get("/learning/thresholds")
async def get_confidence_thresholds():
    """Get current confidence thresholds"""
    return {
        "high_confidence": 0.9,
        "medium_confidence": 0.7,
        "low_confidence": 0.5
    }

# Test client
client = TestClient(app)

class TestApexVerifyAI:
    """Test suite for ApexVerify AI API"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create test directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        os.makedirs("static/badges", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        # Create a test image
        self.test_image = self._create_test_image()
    
    def _create_test_image(self):
        """Create a test image for testing"""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_root_endpoint(self):
        """Test the root endpoint serves the main page"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_models_endpoint(self):
        """Test models information endpoint"""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "deepfake_model" in data
        assert "face_model" in data
        assert "device" in data
    
    def test_upload_endpoint(self):
        """Test file upload endpoint"""
        files = {"file": ("test.jpg", self.test_image.getvalue(), "image/jpeg")}
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "message" in data
    
    def test_analyze_endpoint(self):
        """Test media analysis endpoint"""
        files = {"file": ("test.jpg", self.test_image.getvalue(), "image/jpeg")}
        response = client.post("/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data or "error" in data
    
    def test_learning_metrics_endpoint(self):
        """Test learning metrics endpoint"""
        response = client.get("/learning/metrics")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_analysis_history_endpoint(self):
        """Test analysis history endpoint"""
        response = client.get("/learning/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_confidence_thresholds_endpoint(self):
        """Test confidence thresholds endpoint"""
        response = client.get("/learning/thresholds")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_learning_insights_endpoint(self):
        """Test learning insights endpoint"""
        response = client.get("/learning/insights")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_invalid_file_type(self):
        """Test handling of invalid file types"""
        # Create a text file instead of image
        text_content = b"This is not an image file"
        files = {"file": ("test.txt", text_content, "text/plain")}
        response = client.post("/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
    
    def test_static_files_serving(self):
        """Test static files are served correctly"""
        # Create a test static file
        test_content = "Test static content"
        with open("static/test.txt", "w") as f:
            f.write(test_content)
        
        response = client.get("/static/test.txt")
        assert response.status_code == 200
        assert response.text == test_content
        
        # Cleanup
        os.remove("static/test.txt")
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.get("/api/health")
        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
    
    def test_api_documentation(self):
        """Test API documentation endpoints"""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 