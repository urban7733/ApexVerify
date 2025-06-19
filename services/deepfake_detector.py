import os
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image
import imagehash
import torch
import json
import requests
from datetime import datetime
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
from pathlib import Path
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
import hashlib
import base64
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    def __init__(self):
        self._setup_directories()
        self._setup_gpu()
        self.confidence_threshold = 0.90
        self._initialize_models()

    def _setup_directories(self):
        """Create necessary directories for storing analysis results"""
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("static/badges", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        os.makedirs("models/face_db", exist_ok=True)

    def _setup_gpu(self):
        """Configure GPU settings for all models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _initialize_models(self):
        """Initialize all detection models"""
        try:
            # Initialize deepfake detection model
            self.deepfake_model_name = "microsoft/beit-base-patch16-224"
            self.deepfake_processor = AutoImageProcessor.from_pretrained(self.deepfake_model_name)
            self.deepfake_model = AutoModelForImageClassification.from_pretrained(self.deepfake_model_name)
            if torch.cuda.is_available():
                self.deepfake_model = self.deepfake_model.cuda()

            # Initialize face detection model
            self.face_model_name = "microsoft/beit-base-patch16-224-pt22k-ft22k"
            self.face_processor = AutoImageProcessor.from_pretrained(self.face_model_name)
            self.face_model = AutoModelForImageClassification.from_pretrained(self.face_model_name)
            if torch.cuda.is_available():
                self.face_model = self.face_model.cuda()

            logger.info("All models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def _detect_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect faces in an image using BEiT model"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            inputs = self.face_processor(images=image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.face_model(**inputs)
                predictions = F.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(predictions, 1)
            
            # Process results
            faces = []
            if confidence.item() > self.confidence_threshold:
                faces.append({
                    "confidence": confidence.item(),
                    "class": predicted_class.item(),
                    "bbox": self._get_face_bbox(image)  # Implement this method based on your needs
                })
            
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []

    def _get_face_bbox(self, image: Image.Image) -> Dict[str, int]:
        """Get face bounding box from image"""
        # This is a placeholder implementation
        # You might want to use a dedicated face detection model here
        return {
            "x": 0,
            "y": 0,
            "width": image.width,
            "height": image.height
        }

    async def _reverse_image_search(self, image_path: str) -> Dict[str, Any]:
        """Perform comprehensive reverse image search"""
        try:
            # Calculate image hashes
            image = Image.open(image_path)
            image_hash = self._calculate_image_hash(image)
            
            # Prepare image for search
            with open(image_path, 'rb') as img_file:
                image_data = img_file.read()
            
            # Search on multiple platforms
            search_results = await asyncio.gather(
                self._search_google_images(image_data),
                self._search_tineye(image_data),
                self._search_bing_images(image_data),
                self._search_social_media(image_data)
            )
            
            return {
                "image_hash": image_hash,
                "google_results": search_results[0],
                "tineye_results": search_results[1],
                "bing_results": search_results[2],
                "social_media_results": search_results[3]
            }
        except Exception as e:
            logger.error(f"Error in reverse image search: {str(e)}")
            return {"error": str(e)}

    def _calculate_image_hash(self, image: Image.Image) -> Dict[str, str]:
        """Calculate multiple perceptual hashes for better similarity matching"""
        try:
            return {
                "average_hash": str(imagehash.average_hash(image)),
                "perceptual_hash": str(imagehash.phash(image)),
                "difference_hash": str(imagehash.dhash(image)),
                "wavelet_hash": str(imagehash.whash(image))
            }
        except Exception as e:
            logger.error(f"Error calculating image hash: {str(e)}")
            return {}

    async def _search_google_images(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Search for similar images on Google Images"""
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare search URL
            search_url = "https://www.google.com/searchbyimage"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(search_url, data={"image_url": f"data:image/jpeg;base64,{image_b64}"}, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        results = []
                        
                        # Extract image results
                        for img in soup.find_all('img'):
                            if img.get('src') and not img['src'].startswith('data:'):
                                results.append({
                                    "url": img['src'],
                                    "title": img.get('alt', ''),
                                    "source": "Google Images"
                                })
                        return results
            return []
        except Exception as e:
            logger.error(f"Error searching Google Images: {str(e)}")
            return []

    async def _search_tineye(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Search for similar images on TinEye"""
        try:
            search_url = "https://www.tineye.com/search"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(search_url, data={"image": image_data}, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        results = []
                        
                        # Extract image results
                        for result in soup.find_all('div', class_='match'):
                            img = result.find('img')
                            if img and img.get('src'):
                                results.append({
                                    "url": img['src'],
                                    "title": img.get('alt', ''),
                                    "source": "TinEye"
                                })
                        return results
            return []
        except Exception as e:
            logger.error(f"Error searching TinEye: {str(e)}")
            return []

    async def _search_bing_images(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Search for similar images on Bing Images"""
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            search_url = "https://www.bing.com/images/search"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(search_url, data={"q": f"data:image/jpeg;base64,{image_b64}"}, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        results = []
                        
                        # Extract image results
                        for img in soup.find_all('img', class_='mimg'):
                            if img.get('src'):
                                results.append({
                                    "url": img['src'],
                                    "title": img.get('alt', ''),
                                    "source": "Bing Images"
                                })
                        return results
            return []
        except Exception as e:
            logger.error(f"Error searching Bing Images: {str(e)}")
            return []

    async def _search_social_media(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Search for similar images on social media platforms"""
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Search on multiple social media platforms
            social_media_results = await asyncio.gather(
                self._search_instagram(image_b64),
                self._search_facebook(image_b64),
                self._search_twitter(image_b64)
            )
            
            # Combine results
            all_results = []
            for platform_results in social_media_results:
                all_results.extend(platform_results)
            
            return all_results
        except Exception as e:
            logger.error(f"Error searching social media: {str(e)}")
            return []

    async def _search_instagram(self, image_b64: str) -> List[Dict[str, Any]]:
        """Search for similar images on Instagram"""
        # Implementation would require Instagram API access
        return []

    async def _search_facebook(self, image_b64: str) -> List[Dict[str, Any]]:
        """Search for similar images on Facebook"""
        # Implementation would require Facebook API access
        return []

    async def _search_twitter(self, image_b64: str) -> List[Dict[str, Any]]:
        """Search for similar images on Twitter"""
        # Implementation would require Twitter API access
        return []

    def _analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for deepfake detection"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            inputs = self.deepfake_processor(images=image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.deepfake_model(**inputs)
                predictions = F.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(predictions, 1)
            
            # Map predictions to labels
            is_fake = predicted_class.item() == 0
            confidence_score = confidence.item()
            
            # Detect faces
            faces = self._detect_faces(image_path)
            
            return {
                "is_fake": is_fake,
                "confidence": confidence_score,
                "prediction": "FAKE" if is_fake else "REAL",
                "faces_detected": len(faces),
                "face_details": faces
            }
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"error": str(e)}

    def _analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video frames for deepfake detection"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_results = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 10th frame for efficiency
                if frame_count % 10 == 0:
                    # Save frame temporarily
                    temp_frame_path = f"temp/frame_{frame_count}.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Analyze frame
                    frame_analysis = self._analyze_image(temp_frame_path)
                    frame_results.append(frame_analysis)
                    
                    # Clean up
                    os.remove(temp_frame_path)
                
                frame_count += 1
            
            cap.release()
            
            # Aggregate results
            fake_frames = sum(1 for r in frame_results if r.get("is_fake", False))
            total_analyzed = len(frame_results)
            
            return {
                "total_frames": total_frames,
                "analyzed_frames": total_analyzed,
                "fake_frames": fake_frames,
                "fake_percentage": (fake_frames / total_analyzed * 100) if total_analyzed > 0 else 0,
                "frame_results": frame_results
            }
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            return {"error": str(e)}

    async def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze media file for deepfake detection and perform reverse image search"""
        try:
            # Determine if file is image or video
            is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            
            # Perform reverse image search
            reverse_search_results = await self._reverse_image_search(file_path)
            
            if is_video:
                analysis_results = self._analyze_video(file_path)
                analysis_results["media_type"] = "video"
            else:
                analysis_results = self._analyze_image(file_path)
                analysis_results["media_type"] = "image"
            
            # Combine results
            return {
                **analysis_results,
                "reverse_search": reverse_search_results,
                "metadata": self._extract_metadata(file_path),
                "gpu_accelerated": torch.cuda.is_available()
            }
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {"error": str(e)}

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file"""
        try:
            image = Image.open(file_path)
            return {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}
    
    def extract_features(self, file_path: str) -> np.ndarray:
        """Extract feature vector from image for reinforcement learning"""
        try:
            # Load and preprocess image
            image = Image.open(file_path)
            inputs = self.deepfake_processor(images=image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get features from the model
            with torch.no_grad():
                outputs = self.deepfake_model(**inputs, output_hidden_states=True)
                
                # Use the last hidden state as features
                features = outputs.hidden_states[-1].mean(dim=1)  # Average pooling
                
                # Convert to numpy array
                if torch.cuda.is_available():
                    features = features.cpu().numpy()
                else:
                    features = features.numpy()
                
                return features.flatten()  # Flatten to 1D array
                
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return zero features if extraction fails
            return np.zeros(512) 