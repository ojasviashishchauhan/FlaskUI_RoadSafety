import os
import numpy as np
from PIL import Image
import traceback
from threading import Lock
import cv2
import time # Add time for simple profiling
# import random # No longer needed for dummy boxes
# from flask import current_app # No longer needed here
import logging # Import standard logging

# Uncomment ultralytics import
from ultralytics import YOLO

# Get a logger for this module
logger = logging.getLogger(__name__)

class MLPredictor:
    """Class for making predictions using ML models"""
    
    def __init__(self, models_dir):
        """Initialize predictor with models directory
        
        Args:
            models_dir: Path to directory containing model files
        """
        self.models_dir = models_dir
        self.models = {}
        # self.class_names = ['Crack', 'Pothole', 'Manhole', 'Patch']  # Default class names - seems unused now?
        self.lock = Lock()  # Lock for thread safety
        
        # Map form values to their specific model files - each damage type has its own model
        self.model_files = {
            'Potholes': 'Potholes.pt', 
            'Longitudinal': 'Longitudinal.pt',
            'Transverse': 'Transverse.pt',
            'Alligator': 'Alligator.pt',
            'Edge': 'Edge.pt',
            'Reflection': 'Reflection.pt',
            'All': 'All.pt'
        }
        
        # Use module-level logger
        logger.info(f"Initializing MLPredictor with models directory: {models_dir}")
        
        # List available model files for debugging
        try:
            available_models = os.listdir(models_dir)
            logger.info(f"Available model files: {available_models}")
        except Exception as e:
            logger.error(f"Error listing models directory: {e}")
            available_models = []
        
        # Find the first valid model to use as fallback
        fallback_model = None
        fallback_model_path = None # Keep track of the fallback path
        
        # Try to load Potholes.pt first as a likely fallback
        pothole_path = os.path.join(models_dir, 'Potholes.pt')
        if os.path.exists(pothole_path) and os.path.getsize(pothole_path) > 1000: 
            try:
                logger.info(f"Attempting to load fallback model from {pothole_path}")
                fallback_model = YOLO(pothole_path)
                fallback_model_path = pothole_path
                logger.info(f"Fallback model loaded successfully from {fallback_model_path}")
            except Exception as e:
                logger.error(f"Error loading fallback model from {pothole_path}: {e}")
        
        # Load each model separately
        for damage_type, model_file in self.model_files.items():
            model_path = os.path.join(models_dir, model_file)
            # Skip loading if it's the same as the already loaded fallback
            if fallback_model and model_path == fallback_model_path:
                logger.info(f"Assigning already loaded fallback model for {damage_type}")
                self.models[damage_type] = fallback_model
                continue
            
            try:
                # Check if file exists and isn't just an empty placeholder
                if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
                    logger.info(f"Loading YOLO model for {damage_type} from {model_path}")
                    self.models[damage_type] = YOLO(model_path)
                    logger.info(f"YOLO model for {damage_type} loaded successfully")
                    # If this is the first model loaded and we didn't have a Pothole fallback, use this as fallback
                    if not fallback_model:
                         fallback_model = self.models[damage_type]
                         fallback_model_path = model_path
                         logger.info(f"Using {model_file} as the primary fallback model.")
                else:
                    logger.warning(f"Model file for {damage_type} not found or is too small at {model_path}")
                    if fallback_model:
                        logger.warning(f"Using fallback model ({os.path.basename(fallback_model_path)}) for {damage_type}")
                        self.models[damage_type] = fallback_model
                    else:
                        # This case should ideally not be reached if at least one model exists
                        logger.error(f"No valid model found for {damage_type} and no fallback available!")
                        self.models[damage_type] = None # Assign None if no model can be loaded
            except Exception as e:
                logger.error(f"Error loading YOLO model for {damage_type} from {model_path}: {e}")
                if fallback_model:
                    logger.warning(f"Using fallback model ({os.path.basename(fallback_model_path)}) for {damage_type} after error.")
                    self.models[damage_type] = fallback_model
                else:
                    logger.error(f"No valid model found for {damage_type} after error and no fallback available!")
                    self.models[damage_type] = None # Assign None
        
        if not self.models or all(v is None for v in self.models.values()):
            logger.critical("CRITICAL: No models were loaded successfully, including fallbacks. Predictor may not function.")
        elif not fallback_model:
             logger.warning("Warning: No fallback model could be loaded.")

    def predict(self, image_path, image_type):
        """Make prediction on an image using YOLOv8
        
        Args:
            image_path: Path to image file
            image_type: Type of damage to detect
            
        Returns:
            Dict containing prediction results
        """
        # Use module-level logger (defined above)
        img_basename = os.path.basename(image_path)
        logger.info(f"[Predict START] Received request for image: {img_basename}, type: {image_type}")
        start_time = time.time()
        try:
            # logger.debug(f"[Predict {img_basename}] Attempting to acquire lock...")
            with self.lock:
                # logger.debug(f"[Predict {img_basename}] Lock acquired.")
                
                model = self.models.get(image_type)
                
                # Handle case where model loading failed completely for this type
                if model is None:
                     logger.error(f"[Predict ERROR {img_basename}] No model loaded for type: {image_type}. Cannot predict.")
                     # Return an error structure consistent with success case
                     return {
                         'raw_predictions': [],
                         'annotated_path': None,
                         'error': f"No model available or loaded for type {image_type}"
                     }

                # logger.info(f"[Predict {img_basename}] Running prediction with {image_type} model...")
                model_start = time.time()
                
                # Run YOLOv8 prediction
                # logger.info(f"Using conf=0.2, iou=0.2") # Less verbose
                results = model(image_path, conf=0.2, iou=0.2, verbose=False) # verbose=False reduces console spam
                # logger.info(f"[Predict {img_basename}] Prediction completed")
                
                # Process results
                predictions = self._process_yolo_results(results)
                # logger.info(f"[Predict {img_basename}] Generated {len(predictions)} predictions")
                pred_time = time.time() - model_start
                # logger.info(f"[Predict {img_basename}] Prediction finished. (Took {pred_time:.2f}s)")
                
                # Create an annotated image with predictions
                # logger.info(f"[Predict {img_basename}] Creating annotated image...")
                annotate_start = time.time()
                
                annotated_img = None
                annotated_path = None # Initialize annotated_path
                if results and len(results) > 0 and hasattr(results[0], 'plot'):
                    try:
                        # Use the YOLO plot method to get the annotated image
                        annotated_img = results[0].plot() # Returns BGR numpy array
                    except Exception as plot_err:
                         logger.error(f"[Predict ERROR {img_basename}] results[0].plot() failed: {plot_err}")
                         # Fallback: Load original image if plot fails
                         annotated_img = cv2.imread(image_path)
                else:
                    # Fallback if no results or plot fails
                    logger.warning(f"[Predict WARN {img_basename}] No results or plot method unavailable. Loading original image for annotation attempt.")
                    annotated_img = cv2.imread(image_path)
                
                annotate_time = time.time() - annotate_start
                # logger.info(f"[Predict {img_basename}] Annotation attempt finished. (Took {annotate_time:.2f}s)")
                
                if annotated_img is not None:
                    base_path = os.path.dirname(image_path)
                    filename = os.path.basename(image_path)
                    name, ext = os.path.splitext(filename)
                    annotated_filename = f"{name}_annotated{ext}"
                    annotated_full_path = os.path.join(base_path, annotated_filename)
                    
                    # logger.info(f"[Predict {img_basename}] Saving annotated image to {annotated_filename}...")
                    save_start = time.time()
                    try:
                        # Ensure image is in BGR format for imwrite
                        if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                             # Already BGR presumably (from plot() or imread)
                             pass
                        else:
                             # Handle grayscale or other formats if necessary, or log error
                             logger.warning(f"[Predict WARN {img_basename}] Annotated image is not in expected BGR format. Shape: {annotated_img.shape}")
                             # Attempt conversion if needed, otherwise skip save?
                        
                        save_success = cv2.imwrite(annotated_full_path, annotated_img)
                        if not save_success:
                            logger.error(f"[Predict ERROR {img_basename}] cv2.imwrite failed to save annotated image to {annotated_full_path}.")
                        else:
                            save_time = time.time() - save_start
                            # logger.info(f"[Predict {img_basename}] Annotated image saved. (Took {save_time:.2f}s)")
                            annotated_path = annotated_filename  # Store only filename
                    except Exception as imwrite_err:
                         logger.error(f"[Predict ERROR {img_basename}] Exception during cv2.imwrite: {imwrite_err}")
                else:
                    logger.warning(f"[Predict WARN {img_basename}] No annotated image was created or loaded.")
                    
                total_time = time.time() - start_time
                logger.info(f"[Predict SUCCESS {img_basename}] Prediction successful. (Pred: {pred_time:.2f}s, Annot: {annotate_time:.2f}s, Total: {total_time:.2f}s)")
                return {
                    'raw_predictions': predictions,
                    'annotated_path': annotated_path,
                    'error': None
                }
                
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[Predict ERROR {img_basename}] Exception occurred: {e} (Total time: {total_time:.2f}s)")
            logger.error(traceback.format_exc()) # Log full traceback for errors
            return {
                'raw_predictions': [],
                'annotated_path': None,
                'error': str(e)
            }
        finally:
            # logger.debug(f"[Predict END {img_basename}] Exiting predict method.") # Maybe too verbose
            pass

    def _process_yolo_results(self, results):
        """Convert YOLOv8 results to our standard prediction format
        
        Args:
            results: Results from YOLOv8 prediction
            
        Returns:
            List of prediction dictionaries in our standard format
        """
        predictions = []
        # Use module-level logger (defined above)
        
        if not results or len(results) == 0 or results[0].boxes is None:
            # logger.debug("No results or boxes found to process.")
            return predictions
        
        result = results[0]  # Get first result (should be only one for single image)
        boxes = result.boxes # Access the Boxes object
        
        # Check if we have boxes (detection) or masks (segmentation) results
        if len(boxes) > 0:
            # logger.debug(f"Processing {len(boxes)} detections.")
            # Extract data directly from the Boxes object for efficiency
            bboxes_xyxy = boxes.xyxy.cpu().numpy().tolist() if boxes.xyxy is not None else []
            confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
            cls_ids = boxes.cls.cpu().numpy().tolist() if boxes.cls is not None else []
            class_names_map = result.names # Class ID -> Name mapping

            # Process masks if available
            masks_xy = []
            if result.masks is not None and hasattr(result.masks, 'xy') and len(result.masks.xy) == len(boxes):
                 masks_xy = result.masks.xy # List of numpy arrays
            elif result.masks is not None:
                 logger.warning("Masks object present but format mismatch or 'xy' attribute missing.")

            for i in range(len(bboxes_xyxy)):
                # Get class name
                cls_id = int(cls_ids[i])
                class_name = class_names_map.get(cls_id, f"Class_{cls_id}")
                
                # Create prediction dict
                prediction = {
                    'class_name': class_name,
                    'confidence': float(confs[i]),
                    'bbox': bboxes_xyxy[i]
                }
                
                # Add mask information if available for this index
                if i < len(masks_xy):
                     try:
                         # Mask points are already numpy arrays, convert to list
                         mask_points = masks_xy[i].tolist()
                         prediction['mask'] = mask_points
                     except Exception as mask_err:
                         logger.warning(f"Error processing mask for detection {i}: {mask_err}")
                
                predictions.append(prediction)
        # else: logger.debug("No bounding boxes found in results.")
            
        return predictions

    def predict_frame(self, frame_bgr, image_type='All'):
        """Make prediction on a single video frame using YOLOv8
        
        Args:
            frame_bgr: Frame as NumPy array (in BGR format)
            image_type: Type of damage to detect
            
        Returns:
            List of prediction dictionaries for the frame
        """
        # Use module-level logger (defined above)
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
             logger.warning("predict_frame received invalid frame.")
             return []
             
        model = self.models.get(image_type) 
        if model is None:
             logger.error(f"predict_frame: No model available for type {image_type}.")
             return []
        
        try:
            # Run YOLOv8 prediction on the frame using the appropriate model
            # Use lower confidence/iou for frame processing? Maybe configurable.
            results = model(frame_bgr, conf=0.25, iou=0.45, verbose=False) # Adjusted conf/iou, no verbose
            
            # Process results
            predictions = self._process_yolo_results(results)
            return predictions
        except Exception as e:
            logger.error(f"Error in predict_frame for type {image_type}: {e}")
            logger.error(traceback.format_exc())
            return []

# --- Dummy Model Removed --- 