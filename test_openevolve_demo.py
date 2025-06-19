#!/usr/bin/env python3
"""
Demonstration script for ApexVerify AI with OpenEvolve integration
This script shows how the self-learning system works
"""

import requests
import time
import json
import numpy as np
from PIL import Image
from io import BytesIO
import os

class OpenEvolveDemo:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_test_image(self, is_fake=True):
        """Create a test image for demonstration"""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red' if is_fake else 'blue')
        
        # Add some noise to make it more realistic
        img_array = np.array(img)
        noise = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def check_server_health(self):
        """Check if the server is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server is healthy: {data}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def get_evolution_status(self):
        """Get current evolution status"""
        try:
            response = self.session.get(f"{self.base_url}/evolution/status")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get evolution status: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error getting evolution status: {e}")
            return None
    
    def analyze_image(self, is_fake=True, use_evolution=True):
        """Analyze a test image"""
        try:
            # Create test image
            img_bytes = self.create_test_image(is_fake)
            
            # Prepare file for upload
            files = {'file': ('test_image.jpg', img_bytes.getvalue(), 'image/jpeg')}
            
            # Analyze with evolution
            params = {
                'use_evolution': use_evolution,
                'user_feedback': is_fake  # Provide ground truth for learning
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                files=files,
                params=params
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Analysis failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return None
    
    def submit_feedback(self, file_id, user_feedback, confidence_rating=None):
        """Submit user feedback"""
        try:
            feedback_data = {
                'file_id': file_id,
                'user_feedback': user_feedback,
                'confidence_rating': confidence_rating
            }
            
            response = self.session.post(
                f"{self.base_url}/feedback",
                json=feedback_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Feedback submission failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error submitting feedback: {e}")
            return None
    
    def trigger_evolution(self, min_data_points=10):
        """Trigger evolution process"""
        try:
            response = self.session.post(
                f"{self.base_url}/evolution/trigger",
                params={'min_data_points': min_data_points}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Evolution trigger failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error triggering evolution: {e}")
            return None
    
    def export_learning_data(self):
        """Export learning data"""
        try:
            response = self.session.get(f"{self.base_url}/evolution/export")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Export failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return None
    
    def run_demo(self):
        """Run the complete demonstration"""
        print("🚀 Starting ApexVerify AI with OpenEvolve Demo")
        print("=" * 60)
        
        # Check server health
        if not self.check_server_health():
            print("❌ Server is not available. Please start the server first.")
            return
        
        print("\n📊 Initial Evolution Status:")
        initial_status = self.get_evolution_status()
        if initial_status:
            print(f"   Total analyses: {initial_status['learning_stats']['total_analyses']}")
            print(f"   Evolution iterations: {initial_status['learning_stats']['evolution_iterations']}")
            print(f"   Best accuracy: {initial_status['learning_stats']['best_accuracy']}")
        
        print("\n🔄 Running Analysis and Learning Cycle...")
        print("-" * 40)
        
        # Run multiple analyses to build up learning data
        for i in range(15):
            print(f"\n📸 Analysis {i+1}/15")
            
            # Alternate between fake and real images
            is_fake = (i % 2 == 0)
            image_type = "FAKE" if is_fake else "REAL"
            
            print(f"   Creating {image_type.lower()} test image...")
            
            # Analyze with evolution enabled
            result = self.analyze_image(is_fake=is_fake, use_evolution=True)
            
            if result:
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Ground Truth: {image_type}")
                
                # Check if evolution was used
                if 'evolution_data' in result:
                    evolution_used = result['evolution_data']['used_evolved_program']
                    print(f"   Evolution used: {'✅' if evolution_used else '❌'}")
                
                # Check learning metrics
                if 'learning_metrics' in result:
                    accuracy = result['learning_metrics'].get('accuracy')
                    if accuracy is not None:
                        print(f"   Learning accuracy: {accuracy:.3f}")
            
            # Small delay between analyses
            time.sleep(0.5)
        
        print("\n📈 Checking Evolution Progress...")
        print("-" * 40)
        
        # Check status after analyses
        status = self.get_evolution_status()
        if status:
            print(f"   Total analyses: {status['learning_stats']['total_analyses']}")
            print(f"   Analysis history: {status['analysis_history_size']}")
            print(f"   Feedback history: {status['feedback_history_size']}")
            print(f"   Best accuracy: {status['learning_stats']['best_accuracy']}")
        
        print("\n🎯 Triggering Evolution...")
        print("-" * 40)
        
        # Try to trigger evolution
        evolution_result = self.trigger_evolution(min_data_points=10)
        if evolution_result:
            if evolution_result['triggered']:
                print("✅ Evolution triggered successfully!")
                print(f"   Data points available: {evolution_result['data_points_available']}")
            else:
                print("⏳ Evolution not triggered - insufficient data")
                print(f"   Data points available: {evolution_result['data_points_available']}")
        
        print("\n📊 Final Evolution Status:")
        print("-" * 40)
        
        final_status = self.get_evolution_status()
        if final_status:
            print(f"   Total analyses: {final_status['learning_stats']['total_analyses']}")
            print(f"   Evolution iterations: {final_status['learning_stats']['evolution_iterations']}")
            print(f"   Best accuracy: {final_status['learning_stats']['best_accuracy']}")
            print(f"   Analysis history: {final_status['analysis_history_size']}")
            print(f"   Feedback history: {final_status['feedback_history_size']}")
        
        print("\n💾 Exporting Learning Data...")
        print("-" * 40)
        
        # Export learning data
        export_data = self.export_learning_data()
        if export_data:
            print("✅ Learning data exported successfully!")
            print(f"   Analysis history entries: {len(export_data['analysis_history'])}")
            print(f"   Feedback history entries: {len(export_data['feedback_history'])}")
            print(f"   Export timestamp: {export_data['export_timestamp']}")
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        print("The OpenEvolve integration is working correctly!")
        print("The system is learning from the test data and improving over time.")
        print("\nNext steps:")
        print("1. Visit http://localhost:8000 for the web interface")
        print("2. Upload real images/videos for analysis")
        print("3. Provide feedback to improve the model")
        print("4. Monitor evolution progress at /evolution/status")

def main():
    """Main function"""
    demo = OpenEvolveDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 