import pytest
import asyncio
import aiofiles
import os
import tempfile
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
from io import BytesIO
from main import app

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
        assert "html" in response.headers.get("content-type", "")
    
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
        assert "analysis" in data or "result" in data
    
    def test_batch_analyze_endpoint(self):
        """Test batch analysis endpoint"""
        files = [
            ("files", ("test1.jpg", self.test_image.getvalue(), "image/jpeg")),
            ("files", ("test2.jpg", self.test_image.getvalue(), "image/jpeg"))
        ]
        response = client.post("/api/batch-analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "total_files" in data
        assert "results" in data
    
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
    
    def test_invalid_file_type(self):
        """Test handling of invalid file types"""
        # Create a text file instead of image
        text_content = b"This is not an image file"
        files = {"file": ("test.txt", text_content, "text/plain")}
        response = client.post("/analyze", files=files)
        assert response.status_code == 400
    
    def test_missing_file(self):
        """Test handling of missing file"""
        response = client.post("/analyze")
        assert response.status_code == 422  # Validation error
    
    def test_large_file_handling(self):
        """Test handling of large files"""
        # Create a large test image
        large_img = Image.new('RGB', (2000, 2000), color='blue')
        large_img_bytes = BytesIO()
        large_img.save(large_img_bytes, format='JPEG', quality=95)
        large_img_bytes.seek(0)
        
        files = {"file": ("large_test.jpg", large_img_bytes.getvalue(), "image/jpeg")}
        response = client.post("/analyze", files=files)
        assert response.status_code == 200
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                files = {"file": ("test.jpg", self.test_image.getvalue(), "image/jpeg")}
                response = client.post("/analyze", files=files)
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(status == 200 for status in results)
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with corrupted image data
        corrupted_data = b"corrupted image data"
        files = {"file": ("corrupted.jpg", corrupted_data, "image/jpeg")}
        response = client.post("/analyze", files=files)
        # Should handle gracefully, either 200 with error in response or 500
        assert response.status_code in [200, 500]
    
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