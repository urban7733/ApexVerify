import json
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional
import torch
from torch import nn
import torch.optim as optim
from collections import defaultdict

class LearningService:
    def __init__(self):
        self.learning_dir = Path("data/learning")
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_history_file = self.learning_dir / "analysis_history.json"
        self.model_weights_file = self.learning_dir / "model_weights.pt"
        self.confidence_thresholds_file = self.learning_dir / "confidence_thresholds.json"
        self._init_learning()
        
    def _init_learning(self):
        """Initialize learning data structures"""
        if not self.analysis_history_file.exists():
            self._save_analysis_history({})
        if not self.confidence_thresholds_file.exists():
            self._save_confidence_thresholds({
                "deepfake": 0.5,
                "face_manipulation": 0.5,
                "image_tampering": 0.5
            })
    
    def _load_analysis_history(self) -> Dict:
        """Load analysis history from file"""
        try:
            with open(self.analysis_history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading analysis history: {str(e)}")
            return {}
    
    def _save_analysis_history(self, history: Dict):
        """Save analysis history to file"""
        try:
            with open(self.analysis_history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving analysis history: {str(e)}")
    
    def _load_confidence_thresholds(self) -> Dict:
        """Load confidence thresholds from file"""
        try:
            with open(self.confidence_thresholds_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading confidence thresholds: {str(e)}")
            return {}
    
    def _save_confidence_thresholds(self, thresholds: Dict):
        """Save confidence thresholds to file"""
        try:
            with open(self.confidence_thresholds_file, 'w') as f:
                json.dump(thresholds, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving confidence thresholds: {str(e)}")
    
    def update_analysis_history(self, file_id: str, analysis_result: Dict, user_feedback: Optional[bool] = None):
        """
        Update analysis history with new results and user feedback
        
        Args:
            file_id: Unique identifier for the analyzed file
            analysis_result: Results from the analysis
            user_feedback: Optional user feedback on the analysis (True if correct, False if incorrect)
        """
        try:
            history = self._load_analysis_history()
            
            # Add new analysis to history
            history[file_id] = {
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": analysis_result,
                "user_feedback": user_feedback,
                "features": {
                    "confidence_score": analysis_result.get("confidence_score", 0),
                    "face_analysis": analysis_result.get("face_analysis", {}),
                    "image_hash": analysis_result.get("image_hash", {})
                }
            }
            
            self._save_analysis_history(history)
            self._update_confidence_thresholds(history)
            
        except Exception as e:
            logging.error(f"Error updating analysis history: {str(e)}")
    
    def _update_confidence_thresholds(self, history: Dict):
        """
        Update confidence thresholds based on analysis history and user feedback
        
        Args:
            history: Analysis history dictionary
        """
        try:
            thresholds = self._load_confidence_thresholds()
            
            # Calculate new thresholds based on feedback
            feedback_data = {
                "deepfake": [],
                "face_manipulation": [],
                "image_tampering": []
            }
            
            for file_id, data in history.items():
                if data.get("user_feedback") is not None:
                    analysis = data["analysis"]
                    features = data["features"]
                    
                    # Deepfake confidence
                    feedback_data["deepfake"].append(
                        (features["confidence_score"], data["user_feedback"])
                    )
                    
                    # Face manipulation confidence
                    if "face_analysis" in features:
                        face_conf = features["face_analysis"].get("confidence", 0)
                        feedback_data["face_manipulation"].append(
                            (face_conf, data["user_feedback"])
                        )
                    
                    # Image tampering confidence
                    if "image_hash" in features:
                        hash_conf = features["image_hash"].get("similarity_score", 0)
                        feedback_data["image_tampering"].append(
                            (hash_conf, data["user_feedback"])
                        )
            
            # Update thresholds based on feedback data
            for category, data in feedback_data.items():
                if data:
                    # Calculate optimal threshold using ROC curve
                    confidences, feedbacks = zip(*data)
                    optimal_threshold = self._calculate_optimal_threshold(
                        np.array(confidences),
                        np.array(feedbacks)
                    )
                    thresholds[category] = float(optimal_threshold)
            
            self._save_confidence_thresholds(thresholds)
            
        except Exception as e:
            logging.error(f"Error updating confidence thresholds: {str(e)}")
    
    def _calculate_optimal_threshold(self, confidences: np.ndarray, feedbacks: np.ndarray) -> float:
        """
        Calculate optimal threshold using ROC curve analysis
        
        Args:
            confidences: Array of confidence scores
            feedbacks: Array of user feedback (True/False)
            
        Returns:
            float: Optimal threshold value
        """
        try:
            # Sort by confidence
            sorted_indices = np.argsort(confidences)
            confidences = confidences[sorted_indices]
            feedbacks = feedbacks[sorted_indices]
            
            # Calculate ROC curve
            thresholds = np.unique(confidences)
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in thresholds:
                predictions = confidences >= threshold
                tp = np.sum((predictions == 1) & (feedbacks == 1))
                fp = np.sum((predictions == 1) & (feedbacks == 0))
                fn = np.sum((predictions == 0) & (feedbacks == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            return float(best_threshold)
            
        except Exception as e:
            logging.error(f"Error calculating optimal threshold: {str(e)}")
            return 0.5
    
    def get_confidence_thresholds(self) -> Dict:
        """Get current confidence thresholds"""
        return self._load_confidence_thresholds()
    
    def get_analysis_history(self) -> Dict:
        """Get analysis history"""
        return self._load_analysis_history()
    
    def get_learning_metrics(self) -> Dict:
        """Get learning metrics and statistics"""
        try:
            history = self._load_analysis_history()
            thresholds = self._load_confidence_thresholds()
            
            # Calculate metrics
            total_analyses = len(history)
            feedback_count = sum(1 for data in history.values() if data.get("user_feedback") is not None)
            correct_predictions = sum(1 for data in history.values() 
                                   if data.get("user_feedback") is True)
            
            return {
                "total_analyses": total_analyses,
                "feedback_count": feedback_count,
                "correct_predictions": correct_predictions,
                "accuracy": correct_predictions / feedback_count if feedback_count > 0 else 0,
                "confidence_thresholds": thresholds,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error getting learning metrics: {str(e)}")
            return {} 