#!/usr/bin/env python3
"""
Project Neptune - Person Detection AI Module

This module uses computer vision and deep learning to detect people in flood zones
who may need rescue. It can identify people in images or video streams and
determine if they are in potentially dangerous situations.

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
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PersonDetector")

class PersonDetector:
    """
    Class for detecting people in flood zones who may need rescue
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the person detector
        
        Args:
            model_path: Path to the pre-trained model (if using custom model)
            confidence_threshold: Minimum confidence for detection
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.last_detections = {}
        
        # Initialize OpenCV's DNN module with pre-trained models
        # This is a simpler alternative to TensorFlow for deployment
        try:
            # Load pre-trained model for person detection
            # Using COCO SSD MobileNet for person detection
            weights_path = os.path.join("models", "ssd_mobilenet", "frozen_inference_graph.pb")
            config_path = os.path.join("models", "ssd_mobilenet", "ssd_mobilenet_v2_coco.pbtxt")
            
            if os.path.exists(weights_path) and os.path.exists(config_path):
                self.model = cv2.dnn.readNetFromTensorflow(weights_path, config_path)
                logger.info("Loaded pre-trained SSD MobileNet model for person detection")
            else:
                logger.warning("Pre-trained model files not found. Using OpenCV's HOG detector as fallback.")
                # Fallback to HOG detector
                self.model = cv2.HOGDescriptor()
                self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception as e:
            logger.error(f"Error initializing person detector: {e}")
            logger.warning("Using OpenCV's HOG detector as fallback.")
            # Fallback to HOG detector
            self.model = cv2.HOGDescriptor()
            self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect_persons(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect people in an image
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of dictionaries with detection results
        """
        # Make a copy of the image
        img_copy = image.copy()
        height, width = img_copy.shape[:2]
        
        detections = []
        
        try:
            # Check if we're using OpenCV DNN or HOG
            if isinstance(self.model, cv2.dnn.Net):
                # Use SSD MobileNet
                blob = cv2.dnn.blobFromImage(img_copy, size=(300, 300), swapRB=True, crop=False)
                self.model.setInput(blob)
                output = self.model.forward()
                
                # Process detections
                for i in range(output.shape[2]):
                    confidence = output[0, 0, i, 2]
                    
                    # Filter by confidence and class (person class is 1 in COCO dataset)
                    if confidence > self.confidence_threshold and int(output[0, 0, i, 1]) == 1:
                        # Get bounding box coordinates
                        box = output[0, 0, i, 3:7] * np.array([width, height, width, height])
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Create detection dictionary
                        detection = {
                            "confidence": float(confidence),
                            "bbox": (x1, y1, x2-x1, y2-y1),
                            "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            "timestamp": datetime.now().isoformat()
                        }
                        detections.append(detection)
            else:
                # Use HOG detector
                rects, weights = self.model.detectMultiScale(
                    img_copy, 
                    winStride=(8, 8),
                    padding=(4, 4),
                    scale=1.05
                )
                
                # Process detections
                for i, (x, y, w, h) in enumerate(rects):
                    confidence = weights[i] if len(weights) > i else 0.5
                    
                    if confidence > self.confidence_threshold:
                        # Create detection dictionary
                        detection = {
                            "confidence": float(confidence),
                            "bbox": (x, y, w, h),
                            "center": (int(x + w/2), int(y + h/2)),
                            "timestamp": datetime.now().isoformat()
                        }
                        detections.append(detection)
        
        except Exception as e:
            logger.error(f"Error during person detection: {e}")
        
        logger.info(f"Detected {len(detections)} persons in image")
        return detections
    
    def assess_risk(self, detections: List[Dict[str, Any]], water_level_map: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Assess risk level for detected persons based on water level
        
        Args:
            detections: List of person detections
            water_level_map: Optional water level map (same size as original image)
            
        Returns:
            List of detections with added risk assessment
        """
        # If no water level map is provided, assume moderate risk for all detections
        if water_level_map is None:
            for detection in detections:
                detection["risk_level"] = "moderate"
                detection["needs_rescue"] = True
            return detections
        
        # Assess risk based on water level at person's position
        for detection in detections:
            # Get person's position (feet/bottom of bounding box)
            x, y, w, h = detection["bbox"]
            foot_position = (int(x + w/2), y + h)
            
            # Check if position is within water level map
            if (0 <= foot_position[1] < water_level_map.shape[0] and 
                0 <= foot_position[0] < water_level_map.shape[1]):
                # Get water level at position
                water_level = water_level_map[foot_position[1], foot_position[0]]
                
                # Assess risk
                if water_level > 0.7:  # High water level
                    detection["risk_level"] = "extreme"
                    detection["needs_rescue"] = True
                elif water_level > 0.4:  # Moderate water level
                    detection["risk_level"] = "high"
                    detection["needs_rescue"] = True
                elif water_level > 0.2:  # Low water level
                    detection["risk_level"] = "moderate"
                    detection["needs_rescue"] = True
                else:
                    detection["risk_level"] = "low"
                    detection["needs_rescue"] = False
            else:
                # Default if position is outside water level map
                detection["risk_level"] = "unknown"
                detection["needs_rescue"] = True
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection results on image
        
        Args:
            image: Original image
            detections: List of detection results
            
        Returns:
            Image with drawn detections
        """
        # Make a copy of the image
        img_copy = image.copy()
        
        # Define colors for different risk levels
        colors = {
            "extreme": (0, 0, 255),    # Red
            "high": (0, 69, 255),      # Orange
            "moderate": (0, 215, 255), # Yellow
            "low": (0, 255, 0),        # Green
            "unknown": (255, 0, 255)   # Purple
        }
        
        # Draw each detection
        for detection in detections:
            # Get bounding box
            x, y, w, h = detection["bbox"]
            
            # Get risk level and color
            risk_level = detection.get("risk_level", "unknown")
            color = colors.get(risk_level, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            confidence = detection["confidence"]
            needs_rescue = detection.get("needs_rescue", False)
            label = f"Person: {confidence:.2f}, Risk: {risk_level}"
            if needs_rescue:
                label += " (NEEDS RESCUE)"
            
            cv2.putText(img_copy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_copy
    
    def analyze_video_feed(self, camera_id: str, video_source: str, interval: int = 5, 
                          output_dir: str = "detections") -> None:
        """
        Continuously analyze a video feed for person detection
        
        Args:
            camera_id: Unique identifier for the camera
            video_source: Path to video file or camera index
            interval: Analysis interval in seconds
            output_dir: Directory to save detection images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        logger.info(f"Started person detection on video feed {video_source}")
        
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
                    # Detect persons
                    detections = self.detect_persons(frame)
                    
                    # Assess risk (in a real system, this would use actual water level data)
                    # For demonstration, create a simple water level map
                    height, width = frame.shape[:2]
                    water_level_map = np.zeros((height, width), dtype=np.float32)
                    # Simulate water level increasing from top to bottom
                    for y in range(height):
                        water_level_map[y, :] = y / height
                    
                    detections = self.assess_risk(detections, water_level_map)
                    
                    # Draw detections
                    result_frame = self.draw_detections(frame, detections)
                    
                    # Save frame if persons detected
                    if detections:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(output_dir, f"persons_detected_{camera_id}_{timestamp}.jpg")
                        cv2.imwrite(filename, result_frame)
                        
                        # Log rescue needs
                        rescue_needed = any(d.get("needs_rescue", False) for d in detections)
                        if rescue_needed:
                            logger.warning(f"RESCUE NEEDED: Detected {len(detections)} persons in need of rescue on camera {camera_id}")
                    
                    # Update last analysis time
                    last_analysis_time = current_time
                
                # Small delay to prevent CPU hogging
                time.sleep(0.1)
        
        finally:
            # Release resources
            cap.release()


# Example usage
if __name__ == "__main__":
    # Create detector
    detector = PersonDetector()
    
    # Test with a sample image
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_persons(sample_image)
    
    print("Detection result:", detections)
