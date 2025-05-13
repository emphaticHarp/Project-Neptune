#!/usr/bin/env python3
"""
Project Neptune - Flood Detection AI Module

This module uses computer vision and machine learning to detect flood conditions
from camera feeds or images. It can analyze water levels, detect rapid changes,
and identify flooded areas.

Requirements:
- OpenCV
- TensorFlow/Keras
- NumPy
"""

import os
import cv2
import numpy as np
import time
from datetime import datetime
import logging
from typing import Tuple, List, Dict, Any, Optional

# Optional: TensorFlow import (uncomment if using deep learning model)
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FloodDetector")

class FloodDetector:
    """
    Class for detecting flood conditions from visual data
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the flood detector
        
        Args:
            model_path: Path to the pre-trained model (if using deep learning approach)
        """
        self.model = None
        self.reference_images = {}
        self.last_detections = {}
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            try:
                # For a deep learning approach, uncomment:
                # self.model = load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.info("Using rule-based detection (no model loaded)")
    
    def set_reference_image(self, camera_id: str, image: np.ndarray) -> None:
        """
        Set a reference image for a camera (baseline for comparison)
        
        Args:
            camera_id: Unique identifier for the camera
            image: Reference image as numpy array
        """
        # Convert to grayscale for simpler processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Store reference image
        self.reference_images[camera_id] = gray
        logger.info(f"Reference image set for camera {camera_id}")
    
    def detect_flood(self, camera_id: str, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect flood conditions in an image
        
        Args:
            camera_id: Unique identifier for the camera
            image: Image to analyze as numpy array
            
        Returns:
            Dictionary with detection results
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Check if we have a reference image
        if camera_id not in self.reference_images:
            # If no reference, set this as reference
            self.set_reference_image(camera_id, image)
            return {
                "flood_detected": False,
                "confidence": 0.0,
                "water_level_change": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Get reference image
        reference = self.reference_images[camera_id]
        
        # Calculate difference between current and reference
        diff = cv2.absdiff(gray, reference)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        change_percent = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])
        
        # Detect water level using edge detection
        # This is a simplified approach - in a real system, this would be more sophisticated
        edges = cv2.Canny(gray, 50, 150)
        
        # In the lower half of the image, look for horizontal lines that could be water surface
        height, width = edges.shape
        lower_half = edges[height//2:, :]
        
        # Use Hough transform to detect lines
        lines = cv2.HoughLinesP(lower_half, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        # Calculate water level based on detected lines
        water_level = 0
        if lines is not None:
            # Find the highest horizontal line
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is approximately horizontal
                if abs(y2 - y1) < 10:
                    level = height//2 + min(y1, y2)
                    water_level = max(water_level, level)
        
        # Normalize water level to 0-1 range
        normalized_water_level = water_level / height if height > 0 else 0
        
        # Calculate water level change if we have previous detection
        water_level_change = 0
        if camera_id in self.last_detections:
            prev_level = self.last_detections[camera_id].get("water_level", 0)
            water_level_change = normalized_water_level - prev_level
        
        # Determine if flood is detected based on change and water level
        # This is a simplified rule-based approach
        flood_detected = (change_percent > 0.2 and water_level_change > 0.05) or normalized_water_level > 0.7
        
        # Calculate confidence (simplified)
        confidence = min(1.0, max(0.0, change_percent + abs(water_level_change) * 5))
        
        # Create result dictionary
        result = {
            "flood_detected": flood_detected,
            "confidence": confidence,
            "water_level": normalized_water_level,
            "water_level_change": water_level_change,
            "change_percent": change_percent,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store detection for future reference
        self.last_detections[camera_id] = result
        
        # Log if flood detected
        if flood_detected:
            logger.warning(f"Flood detected by camera {camera_id} with confidence {confidence:.2f}")
        
        return result
    
    def analyze_video_feed(self, camera_id: str, video_source: str, interval: int = 5) -> None:
        """
        Continuously analyze a video feed for flood detection
        
        Args:
            camera_id: Unique identifier for the camera
            video_source: Path to video file or camera index
            interval: Analysis interval in seconds
        """
        # Open video capture
        try:
            if isinstance(video_source, str) and video_source.isdigit():
                cap = cv2.VideoCapture(int(video_source))
            else:
                cap = cv2.VideoCapture(video_source)
        except Exception as e:
            logger.error(f"Error opening video source {video_source}: {e}")
            return
        
        if not cap.isOpened():
            logger.error(f"Could not open video source {video_source}")
            return
        
        logger.info(f"Started flood detection on video feed {video_source}")
        
        last_analysis_time = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"End of video stream for {camera_id}")
                    break
                
                # Check if it's time to analyze
                current_time = time.time()
                if current_time - last_analysis_time >= interval:
                    # Analyze frame
                    result = self.detect_flood(camera_id, frame)
                    
                    # Update last analysis time
                    last_analysis_time = current_time
                    
                    # Process result (in a real system, this would trigger alerts, etc.)
                    if result["flood_detected"]:
                        # Save frame for reference
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"flood_detected_{camera_id}_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                
                # Small delay to prevent CPU hogging
                time.sleep(0.1)
        
        finally:
            # Release resources
            cap.release()
    
    def update_model(self, new_model_path: str) -> bool:
        """
        Update the detection model
        
        Args:
            new_model_path: Path to the new model
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Load new model
            # For a deep learning approach, uncomment:
            # new_model = load_model(new_model_path)
            # self.model = new_model
            
            logger.info(f"Model updated from {new_model_path}")
            return True
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create detector
    detector = FloodDetector()
    
    # Test with a sample image
    sample_image = np.zeros((480, 640), dtype=np.uint8)
    result = detector.detect_flood("test_camera", sample_image)
    
    print("Detection result:", result)
