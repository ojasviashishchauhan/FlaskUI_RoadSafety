import os
import numpy as np
from PIL import Image
import traceback
from threading import Lock
import cv2
import time # Add time for simple profiling
import random # Import for random box generation

# Uncomment ultralytics import
from ultralytics import YOLO

class MLPredictor:
    """Class for making predictions using ML models"""
    
    def __init__(self, models_dir):
        """Initialize predictor with models directory
        
        Args:
            models_dir: Path to directory containing model files
        """
        self.models_dir = models_dir
        self.models = {}
        self.class_names = ['Crack', 'Pothole', 'Manhole', 'Patch']  # Default class names
        self.lock = Lock()  # Lock for thread safety
        
        # Map form values to their specific model files - each damage type has its own model
        # Only include models that actually exist in the models directory
        self.model_files = {
            'Potholes': 'Potholes.pt',  # Correct filename case
            'Longitudinal': 'Longitudinal.pt',
            'Transverse': 'Transverse.pt',
            'Alligator': 'Alligator.pt',
            'Edge': 'Edge.pt',
            'Reflection': 'Reflection.pt',
            'All': 'All.pt'
        }
        
        print(f"Initializing MLPredictor with models directory: {models_dir}")
        # List available model files for debugging
        try:
            available_models = os.listdir(models_dir)
            print(f"Available model files: {available_models}")
        except Exception as e:
            print(f"Error listing models directory: {e}")
            available_models = []
        
        # Find the first valid model to use as fallback
        fallback_model = None
        
        # First try to load Potholes.pt as it's most likely to be a valid model
        pothole_path = os.path.join(models_dir, 'Potholes.pt')
        if os.path.exists(pothole_path) and os.path.getsize(pothole_path) > 1000:  # Check if file exists and isn't empty
            try:
                print(f"Loading fallback model from {pothole_path}")
                fallback_model = YOLO(pothole_path)
                print(f"Fallback model loaded successfully")
            except Exception as e:
                print(f"Error loading fallback model: {e}")
        
        # Load each model separately
        for damage_type, model_file in self.model_files.items():
            model_path = os.path.join(models_dir, model_file)
            try:
                # Check if file exists and isn't just an empty placeholder
                if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
                    print(f"Loading YOLO model for {damage_type} from {model_path}")
                    self.models[damage_type] = YOLO(model_path)
                    print(f"YOLO model for {damage_type} loaded successfully")
                else:
                    print(f"Model file for {damage_type} not found or is too small at {model_path}")
                    if fallback_model:
                        print(f"Using fallback model for {damage_type}")
                        self.models[damage_type] = fallback_model
                    else:
                        print(f"Using dummy model for {damage_type}")
                        self.models[damage_type] = DummyModel(damage_type)
            except Exception as e:
                print(f"Error loading YOLO model for {damage_type}: {e}")
                if fallback_model:
                    print(f"Using fallback model for {damage_type} after error")
                    self.models[damage_type] = fallback_model
                else:
                    print(f"Using dummy model for {damage_type} after error")
                    self.models[damage_type] = DummyModel(damage_type)
        
        if not self.models:
            print("Warning: No models were loaded successfully")
    
    def predict(self, image_path, image_type):
        """Make prediction on an image using YOLOv8
        
        Args:
            image_path: Path to image file
            image_type: Type of damage to detect
            
        Returns:
            Dict containing prediction results
        """
        print(f"[Predict START] Received request for image: {os.path.basename(image_path)}, type: {image_type}")
        start_time = time.time()
        try:
            print(f"[Predict {os.path.basename(image_path)}] Attempting to acquire lock...")
            with self.lock:
                print(f"[Predict {os.path.basename(image_path)}] Lock acquired.")
                if image_type not in self.models:
                    print(f"[Predict ERROR {os.path.basename(image_path)}] No model for type: {image_type}")
                    raise ValueError(f"No model available for {image_type}")
                
                model = self.models[image_type]
                print(f"[Predict {os.path.basename(image_path)}] Running YOLO prediction with {image_type} model...")
                model_start = time.time()
                
                # Run YOLOv8 prediction
                results = model(image_path, conf=0.5, iou=0.5, verbose=False)
                print(f"[Predict {os.path.basename(image_path)}] YOLO prediction completed")
                
                # Process results
                predictions = self._process_yolo_results(results)
                print(f"[Predict {os.path.basename(image_path)}] Generated {len(predictions)} predictions")
                print(f"[Predict {os.path.basename(image_path)}] Prediction finished. (Took {time.time() - model_start:.2f}s)")
                
                # Create an annotated image with predictions
                print(f"[Predict {os.path.basename(image_path)}] Creating annotated image...")
                annotate_start = time.time()
                
                # Get the annotated image from YOLOv8 results
                annotated_img = None
                if results and len(results) > 0:
                    # Use the YOLO plot method to get the annotated image
                    annotated_img = results[0].plot()
                else:
                    # Fallback if no results
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Create a simple annotated version
                        annotated_img = self._create_dummy_annotated_image(img_rgb, image_type)
                    else:
                        print(f"[Predict ERROR {os.path.basename(image_path)}] Failed to read image for annotation")
                
                print(f"[Predict {os.path.basename(image_path)}] Annotation created. (Took {time.time() - annotate_start:.2f}s)")
                
                if annotated_img is not None:
                    base_path = os.path.dirname(image_path)
                    filename = os.path.basename(image_path)
                    name, ext = os.path.splitext(filename)
                    annotated_filename = f"{name}_annotated{ext}"
                    annotated_full_path = os.path.join(base_path, annotated_filename)
                    
                    print(f"[Predict {os.path.basename(image_path)}] Saving annotated image to {annotated_filename}...")
                    save_start = time.time()
                    save_success = cv2.imwrite(annotated_full_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                    if not save_success:
                        print(f"[Predict ERROR {os.path.basename(image_path)}] cv2.imwrite failed to save annotated image.")
                    else:
                        print(f"[Predict {os.path.basename(image_path)}] Annotated image saved. (Took {time.time() - save_start:.2f}s)")
                        annotated_path = annotated_filename  # Store only filename
                else:
                    print(f"[Predict WARN {os.path.basename(image_path)}] No annotated image was created.")
                    annotated_path = None

                total_time = time.time() - start_time
                print(f"[Predict SUCCESS {os.path.basename(image_path)}] YOLO prediction successful. (Total time: {total_time:.2f}s)")
                return {
                    'raw_predictions': predictions,
                    'annotated_path': annotated_path,
                    'error': None
                }
                
        except Exception as e:
            total_time = time.time() - start_time
            print(f"[Predict ERROR {os.path.basename(image_path)}] Exception occurred: {e} (Total time: {total_time:.2f}s)")
            traceback.print_exc()
            return {
                'raw_predictions': [],
                'annotated_path': None,
                'error': str(e)
            }
        finally:
            print(f"[Predict END {os.path.basename(image_path)}] Exiting predict method.")

    def _process_yolo_results(self, results):
        """Convert YOLOv8 results to our standard prediction format
        
        Args:
            results: Results from YOLOv8 prediction
            
        Returns:
            List of prediction dictionaries in our standard format
        """
        predictions = []
        
        if not results or len(results) == 0:
            return predictions
        
        result = results[0]  # Get first result (should be only one for single image)
        
        # Check if we have boxes (for detection) or masks (for segmentation)
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            # Detection results - process bounding boxes
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Extract values - box format is [x1, y1, x2, y2]
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Get class name from model classes if available
                if hasattr(result, 'names') and cls_id in result.names:
                    class_name = result.names[cls_id]
                else:
                    class_name = f"Class_{cls_id}"
                
                # Create prediction dict
                prediction = {
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': bbox
                }
                
                # Add mask information if available
                if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > i:
                    try:
                        # Try to extract mask points
                        mask_points = result.masks.xy[i].tolist()
                        prediction['mask'] = mask_points
                    except (AttributeError, IndexError) as e:
                        # If mask extraction fails, log and continue without mask
                        print(f"Error extracting mask for prediction {i}: {e}")
                
                predictions.append(prediction)
        
        return predictions

    def predict_frame(self, frame_rgb, image_type='All'):
        """Make prediction on a single video frame using YOLOv8
        
        Args:
            frame_rgb: Frame as NumPy array (in RGB format)
            image_type: Type of damage to detect
            
        Returns:
            List of prediction dictionaries for the frame
        """
        if not frame_rgb.size or image_type not in self.models:
            return []
        
        try:
            # Run YOLOv8 prediction on the frame using the appropriate model
            model = self.models[image_type]
            results = model(frame_rgb, conf=0.5, iou=0.5, verbose=False)
            
            # Process results
            predictions = self._process_yolo_results(results)
            return predictions
        except Exception as e:
            print(f"Error in predict_frame: {e}")
            return []

    def _create_dummy_annotated_image(self, image, image_type):
        """Create a simple annotated version of the image for dummy mode
        
        Args:
            image: Original image
            image_type: Type of damage detection being performed
            
        Returns:
            Annotated image
        """
        # Add a green border
        border_size = 20
        h, w = image.shape[:2]
        
        # Draw a border around the image
        cv2.rectangle(image, (0, 0), (w, h), (0, 255, 0), border_size)
        
        # Add text at the top
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Analysis: {image_type}"
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        
        # Add text background
        cv2.rectangle(image, 
                      (w//2 - text_size[0]//2 - 10, 40), 
                      (w//2 + text_size[0]//2 + 10, 40 + text_size[1] + 20),
                      (0, 0, 0), -1)
        
        # Add text
        cv2.putText(image, text, 
                   (w//2 - text_size[0]//2, 40 + text_size[1]), 
                   font, 1.5, (255, 255, 255), 3)
        
        # Add timestamp
        timestamp = f"Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ts_size = cv2.getTextSize(timestamp, font, 0.7, 2)[0]
        
        # Add text background for timestamp
        cv2.rectangle(image, 
                      (w - ts_size[0] - 20, h - ts_size[1] - 20), 
                      (w - 10, h - 10),
                      (0, 0, 0), -1)
        
        # Add timestamp text
        cv2.putText(image, timestamp, 
                   (w - ts_size[0] - 15, h - 15), 
                   font, 0.7, (255, 255, 255), 2)
        
        return image


class DummyModel:
    """Dummy model class that returns empty results"""
    
    def __init__(self, model_type):
        self.model_type = model_type
        
    def __call__(self, image, conf=0.25, iou=0.45, verbose=True):
        """Dummy prediction method"""
        # Create a dummy results structure that mimics what a YOLO model returns
        return [DummyResults()]


class DummyResults:
    """Dummy results class that mimics the structure of ultralytics Results"""
    
    def __init__(self):
        self.boxes = []
        self.masks = None
        
    def plot(self):
        """Dummy plot method that returns a blank image"""
        # Create a blank image (100x100 black image)
        return np.zeros((100, 100, 3), dtype=np.uint8) 