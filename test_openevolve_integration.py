import pytest
import os
import tempfile
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
import json
from unittest.mock import Mock, patch

# Import our services
from services.openevolve_integration import OpenEvolveDeepfakeDetector, DeepfakeAnalysisResult
from main import app

# Test client
client = TestClient(app)

class TestOpenEvolveIntegration:
    """Test suite for OpenEvolve integration"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create test directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        os.makedirs("evolution_output", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Create a test image
        self.test_image = self._create_test_image()
        
        # Initialize OpenEvolve detector
        self.evolution_detector = OpenEvolveDeepfakeDetector(
            evolution_dir="evolution_output",
            enable_evolution=True
        )
    
    def _create_test_image(self):
        """Create a test image for testing"""
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_openevolve_detector_initialization(self):
        """Test OpenEvolve detector initialization"""
        detector = OpenEvolveDeepfakeDetector()
        assert detector.evolution_dir.exists()
        assert detector.enable_evolution is True
    
    def test_analyze_with_evolution(self):
        """Test analysis with evolution enabled"""
        # Create test features
        features = np.random.rand(512)
        metadata = {
            "file_id": "test123",
            "media_type": "image",
            "file_size": 100000
        }
        
        # Test analysis
        result = self.evolution_detector.analyze_with_evolution(
            features=features,
            metadata=metadata,
            ground_truth=True
        )
        
        assert isinstance(result, DeepfakeAnalysisResult)
        assert hasattr(result, 'is_fake')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'prediction')
        assert hasattr(result, 'learning_metrics')
        assert hasattr(result, 'evolution_data')
    
    def test_evolution_status(self):
        """Test evolution status endpoint"""
        response = client.get("/evolution/status")
        assert response.status_code == 200
        data = response.json()
        assert "evolution_enabled" in data
        assert "learning_stats" in data
    
    def test_trigger_evolution(self):
        """Test evolution triggering"""
        # Add some test data
        for i in range(60):
            features = np.random.rand(512)
            metadata = {"file_id": f"test{i}", "media_type": "image"}
            self.evolution_detector.analyze_with_evolution(
                features=features,
                metadata=metadata,
                ground_truth=i % 2 == 0  # Alternate true/false
            )
        
        # Test triggering evolution
        success = self.evolution_detector.trigger_evolution(min_data_points=50)
        assert success is True
    
    def test_trigger_evolution_insufficient_data(self):
        """Test evolution triggering with insufficient data"""
        success = self.evolution_detector.trigger_evolution(min_data_points=100)
        assert success is False
    
    def test_add_feedback(self):
        """Test adding user feedback"""
        self.evolution_detector.add_feedback(
            file_id="test123",
            user_feedback=True,
            confidence_rating=0.8
        )
        
        assert len(self.evolution_detector.feedback_history) == 1
        feedback = self.evolution_detector.feedback_history[0]
        assert feedback["file_id"] == "test123"
        assert feedback["user_feedback"] is True
        assert feedback["confidence_rating"] == 0.8
    
    def test_export_learning_data(self):
        """Test exporting learning data"""
        # Add some test data
        features = np.random.rand(512)
        metadata = {"file_id": "test123", "media_type": "image"}
        self.evolution_detector.analyze_with_evolution(
            features=features,
            metadata=metadata,
            ground_truth=True
        )
        
        self.evolution_detector.add_feedback("test123", True, 0.8)
        
        # Export data
        export_data = self.evolution_detector.export_learning_data()
        
        assert "learning_stats" in export_data
        assert "analysis_history" in export_data
        assert "feedback_history" in export_data
        assert "evolution_status" in export_data
        assert "export_timestamp" in export_data
    
    def test_get_evolution_status(self):
        """Test getting evolution status"""
        status = self.evolution_detector.get_evolution_status()
        
        assert "evolution_enabled" in status
        assert "openevolve_available" in status
        assert "learning_stats" in status
        assert "analysis_history_size" in status
        assert "feedback_history_size" in status
        assert "evolution_directory" in status
        assert "recent_evolutions" in status
    
    def test_analyze_endpoint_with_evolution(self):
        """Test the analyze endpoint with evolution enabled"""
        files = {"file": ("test.jpg", self.test_image.getvalue(), "image/jpeg")}
        response = client.post("/analyze?use_evolution=true", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data or "error" in data
    
    def test_analyze_endpoint_without_evolution(self):
        """Test the analyze endpoint with evolution disabled"""
        files = {"file": ("test.jpg", self.test_image.getvalue(), "image/jpeg")}
        response = client.post("/analyze?use_evolution=false", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data or "error" in data
    
    def test_evolution_trigger_endpoint(self):
        """Test the evolution trigger endpoint"""
        response = client.post("/evolution/trigger?min_data_points=10")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "triggered" in data
        assert "data_points_available" in data
    
    def test_evolution_export_endpoint(self):
        """Test the evolution export endpoint"""
        response = client.get("/evolution/export")
        assert response.status_code == 200
        data = response.json()
        assert "learning_stats" in data
        assert "analysis_history" in data
        assert "feedback_history" in data
    
    def test_feedback_with_evolution(self):
        """Test feedback submission with evolution integration"""
        # First upload and analyze a file
        files = {"file": ("test.jpg", self.test_image.getvalue(), "image/jpeg")}
        upload_response = client.post("/upload", files=files)
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        
        # Submit feedback
        feedback_data = {
            "file_id": file_id,
            "user_feedback": True,
            "confidence_rating": 0.9
        }
        response = client.post("/feedback", json=feedback_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_learning_metrics_integration(self):
        """Test that learning metrics are included in analysis results"""
        features = np.random.rand(512)
        metadata = {"file_id": "test123", "media_type": "image"}
        
        result = self.evolution_detector.analyze_with_evolution(
            features=features,
            metadata=metadata,
            ground_truth=True
        )
        
        assert "learning_metrics" in result.__dict__
        learning_metrics = result.learning_metrics
        assert "confidence" in learning_metrics
        assert "accuracy" in learning_metrics
    
    def test_evolution_data_in_result(self):
        """Test that evolution data is included in analysis results"""
        features = np.random.rand(512)
        metadata = {"file_id": "test123", "media_type": "image"}
        
        result = self.evolution_detector.analyze_with_evolution(
            features=features,
            metadata=metadata,
            ground_truth=True
        )
        
        assert "evolution_data" in result.__dict__
        evolution_data = result.evolution_data
        assert "used_evolved_program" in evolution_data
    
    @patch('services.openevolve_integration.OPENEVOLVE_AVAILABLE', False)
    def test_openevolve_unavailable(self):
        """Test behavior when OpenEvolve is not available"""
        detector = OpenEvolveDeepfakeDetector()
        assert detector.enable_evolution is False
        
        features = np.random.rand(512)
        metadata = {"file_id": "test123", "media_type": "image"}
        
        result = detector.analyze_with_evolution(
            features=features,
            metadata=metadata,
            ground_truth=True
        )
        
        assert isinstance(result, DeepfakeAnalysisResult)
        assert result.evolution_data["used_evolved_program"] is False
    
    def test_evolution_directory_creation(self):
        """Test that evolution directory is created properly"""
        test_dir = "test_evolution_output"
        detector = OpenEvolveDeepfakeDetector(evolution_dir=test_dir)
        
        assert os.path.exists(test_dir)
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
    
    def test_learning_statistics_tracking(self):
        """Test that learning statistics are properly tracked"""
        # Add some analyses
        for i in range(5):
            features = np.random.rand(512)
            metadata = {"file_id": f"test{i}", "media_type": "image"}
            self.evolution_detector.analyze_with_evolution(
                features=features,
                metadata=metadata,
                ground_truth=i % 2 == 0
            )
        
        stats = self.evolution_detector.learning_stats
        assert stats["total_analyses"] == 5
        assert "best_accuracy" in stats
        assert "accuracy_improvements" in stats
    
    def test_analysis_history_management(self):
        """Test that analysis history is properly managed"""
        # Add analyses
        for i in range(10):
            features = np.random.rand(512)
            metadata = {"file_id": f"test{i}", "media_type": "image"}
            self.evolution_detector.analyze_with_evolution(
                features=features,
                metadata=metadata,
                ground_truth=True
            )
        
        # Check history size
        assert len(self.evolution_detector.analysis_history) == 10
        
        # Add more to test limit
        for i in range(1000):
            features = np.random.rand(512)
            metadata = {"file_id": f"test{i}", "media_type": "image"}
            self.evolution_detector.analyze_with_evolution(
                features=features,
                metadata=metadata,
                ground_truth=True
            )
        
        # Should be limited to 500
        assert len(self.evolution_detector.analysis_history) <= 500

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 