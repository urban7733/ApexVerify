import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Optional

class StorageService:
    def __init__(self):
        self.metadata_file = Path("data/metadata.json")
        self.analysis_file = Path("data/analysis.json")
        self.feedback_file = Path("data/feedback.json")
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage files and directories"""
        try:
            # Create data directory
            Path("data").mkdir(exist_ok=True)
            
            # Initialize metadata file if it doesn't exist
            if not self.metadata_file.exists():
                with open(self.metadata_file, "w") as f:
                    json.dump({}, f)
            
            # Initialize analysis file if it doesn't exist
            if not self.analysis_file.exists():
                with open(self.analysis_file, "w") as f:
                    json.dump({}, f)
            
            # Initialize feedback file if it doesn't exist
            if not self.feedback_file.exists():
                with open(self.feedback_file, "w") as f:
                    json.dump({}, f)
                    
        except Exception as e:
            logging.error(f"Error initializing storage: {str(e)}")
            raise
    
    def _read_json(self, file_path: Path) -> Dict:
        """Read JSON data from file"""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {str(e)}")
            return {}
    
    def _write_json(self, file_path: Path, data: Dict):
        """Write JSON data to file"""
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error writing to {file_path}: {str(e)}")
            raise
    
    def store_metadata(self, metadata: Dict):
        """Store file metadata"""
        try:
            data = self._read_json(self.metadata_file)
            file_id = metadata["file_id"]
            data[file_id] = metadata
            self._write_json(self.metadata_file, data)
        except Exception as e:
            logging.error(f"Error storing metadata: {str(e)}")
            raise
    
    def get_metadata(self, file_id: str) -> Optional[Dict]:
        """Retrieve file metadata"""
        try:
            data = self._read_json(self.metadata_file)
            return data.get(file_id)
        except Exception as e:
            logging.error(f"Error retrieving metadata: {str(e)}")
            return None
    
    def store_analysis(self, file_id: str, analysis: Dict):
        """Store analysis results"""
        try:
            data = self._read_json(self.analysis_file)
            data[file_id] = {
                **analysis,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            self._write_json(self.analysis_file, data)
        except Exception as e:
            logging.error(f"Error storing analysis: {str(e)}")
            raise
    
    def get_analysis(self, file_id: str) -> Optional[Dict]:
        """Retrieve analysis results"""
        try:
            data = self._read_json(self.analysis_file)
            return data.get(file_id)
        except Exception as e:
            logging.error(f"Error retrieving analysis: {str(e)}")
            return None
    
    def store_feedback(self, file_id: str, feedback: Dict):
        """Store user feedback"""
        try:
            data = self._read_json(self.feedback_file)
            if file_id not in data:
                data[file_id] = []
            data[file_id].append(feedback)
            self._write_json(self.feedback_file, data)
        except Exception as e:
            logging.error(f"Error storing feedback: {str(e)}")
            raise
    
    def get_feedback(self, file_id: str) -> Optional[list]:
        """Retrieve user feedback for a file"""
        try:
            data = self._read_json(self.feedback_file)
            return data.get(file_id, [])
        except Exception as e:
            logging.error(f"Error retrieving feedback: {str(e)}")
            return None
    
    def get_all_feedback(self) -> Dict:
        """Retrieve all feedback data"""
        try:
            return self._read_json(self.feedback_file)
        except Exception as e:
            logging.error(f"Error retrieving all feedback: {str(e)}")
            return {} 