import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from .ml_predictor import MLPredictor

class DamageDetector:
    """Class for detecting damage in images/video frames"""
    
    def __init__(self, ml_predictor: MLPredictor):
        """Initialize damage detector with ML predictor
        
        Args:
            ml_predictor: Instance of MLPredictor for damage detection
        """
        self.ml_predictor = ml_predictor
        self.confidence_threshold = 0.5  # Minimum confidence for damage detection
        
    def detect(self, image: np.ndarray) -> Dict:
        """Detect damage in an image
        
        Args:
            image: Image as numpy array in RGB format
            
        Returns:
            Dict containing detection results:
                - damage_detected (bool): Whether damage was detected
                - confidence (float): Confidence score of detection
                - damage_type (str): Type of damage detected
                - bounding_boxes (List): List of bounding box coordinates
        """
        try:
            # Run prediction using ML predictor
            prediction = self.ml_predictor.predict_frame(image)
            
            # Get the most confident damage type from detection classes
            damage_classes = prediction.get('raw_predictions', [])
            damage_type = 'No Damage'
            highest_confidence = 0
            bounding_boxes = []
            
            if damage_classes:
                # Sort by confidence and get the highest confidence class
                damage_classes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                highest_pred = damage_classes[0]
                damage_type = highest_pred.get('class_name', 'Unknown')
                highest_confidence = highest_pred.get('confidence', 0)
                
                # Get bounding boxes for all detections above threshold
                bounding_boxes = [
                    pred['bbox'] for pred in damage_classes 
                    if pred.get('confidence', 0) >= self.confidence_threshold
                ]
            
            # Determine if damage is detected based on confidence threshold
            is_damage_detected = highest_confidence >= self.confidence_threshold
            
            return {
                'damage_detected': is_damage_detected,
                'confidence': highest_confidence,
                'damage_type': damage_type,
                'bounding_boxes': bounding_boxes
            }
            
        except Exception as e:
            print(f"Error in damage detection: {str(e)}")
            return {
                'damage_detected': False,
                'confidence': 0.0,
                'damage_type': 'Error',
                'bounding_boxes': [],
                'error': str(e)
            }
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for damage detection
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0") 