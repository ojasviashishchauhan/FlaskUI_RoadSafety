import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import traceback
from threading import Lock
import cv2
import time # Add time for simple profiling

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
        
        # Map form values to actual model files
        self.model_files = {
            'Potholes': 'Potholes.pt',
            'Longitudinal': 'Longitudinal.pt',
            'Transverse': 'Transverse.pt',
            'Alligator': 'Alligator.pt'
        }
        
        # Load all available models
        for damage_type, model_file in self.model_files.items():
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    print(f"Loading model for {damage_type} from {model_path}")
                    self.models[damage_type] = YOLO(model_path)
                    print(f"Successfully loaded model for {damage_type}")
                except Exception as e:
                    print(f"Error loading model for {damage_type}: {e}")
                    traceback.print_exc()
            else:
                print(f"Warning: Model file for {damage_type} not found at {model_path}")
        
        if not self.models:
            print("Warning: No models were loaded successfully")

    def predict(self, image_path, image_type):
        """Make prediction on an image
        
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
                
                print(f"[Predict {os.path.basename(image_path)}] Reading image...")
                read_start = time.time()
                image = cv2.imread(image_path)
                if image is None:
                    print(f"[Predict ERROR {os.path.basename(image_path)}] cv2.imread failed to read image.")
                    raise ValueError(f"Could not read image file: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(f"[Predict {os.path.basename(image_path)}] Image read successfully. Shape: {image.shape} (Took {time.time() - read_start:.2f}s)")
                
                model = self.models[image_type]
                print(f"[Predict {os.path.basename(image_path)}] Running model prediction...")
                model_start = time.time()
                results = model(image, conf=0.25)  # Lower confidence threshold for better recall
                print(f"[Predict {os.path.basename(image_path)}] Model prediction finished. (Took {time.time() - model_start:.2f}s)")
                
                # --- DETAILED LOGGING FOR POTHOLES START ---
                if image_type == 'Potholes':
                    try:
                        print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] Raw results type: {type(results)}")
                        if isinstance(results, list) and results:
                            print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] Number of result objects: {len(results)}")
                            first_result = results[0]
                            print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] First result object type: {type(first_result)}")
                            # Check for boxes
                            if hasattr(first_result, 'boxes') and first_result.boxes:
                                print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] Boxes found: Count={len(first_result.boxes)}")
                            else:
                                print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] No boxes found.")
                            # Check for masks
                            if hasattr(first_result, 'masks') and first_result.masks:
                                print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] Masks attribute found.")
                                if hasattr(first_result.masks, 'xy') and first_result.masks.xy:
                                     print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] Masks.xy found: Count={len(first_result.masks.xy)}")
                                     # Log details of the first mask polygon
                                     if len(first_result.masks.xy) > 0:
                                         first_mask_points = first_result.masks.xy[0]
                                         print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] First mask polygon: Type={type(first_mask_points)}, Points={len(first_mask_points)}")
                                else:
                                     print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] Masks.xy is empty or missing.")
                            else:
                                 print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] No masks attribute found.")
                        else:
                             print(f"-- [Potholes DEBUG {os.path.basename(image_path)}] Results are empty or not a list.")
                    except Exception as debug_err:
                         print(f"-- [Potholes DEBUG ERROR {os.path.basename(image_path)}] Error inspecting results: {debug_err}")
                # --- DETAILED LOGGING FOR POTHOLES END ---
                
                print(f"[Predict {os.path.basename(image_path)}] Processing results...")
                predictions = []
                process_start = time.time()
                for i_res, result in enumerate(results):
                    print(f"[Predict {os.path.basename(image_path)}] Processing result object {i_res+1}/{len(results)}")
                    boxes = result.boxes
                    masks = getattr(result, 'masks', None)
                    has_masks = masks and masks.xy is not None and len(masks.xy) == len(boxes)

                    for i_box, box in enumerate(boxes): # Use enumerate to index masks if they exist
                        pred = {
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': float(box.conf),
                            'class_name': self.class_names[int(box.cls)]
                        }
                        if has_masks:
                            pred['mask'] = masks.xy[i_box].tolist() # Convert numpy array to list
                        predictions.append(pred)
                print(f"[Predict {os.path.basename(image_path)}] Result processing finished. Found {len(predictions)} predictions. (Took {time.time() - process_start:.2f}s)")
                
                # Sort predictions by confidence
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                annotated_path = None
                if predictions:
                    print(f"[Predict {os.path.basename(image_path)}] Creating annotated image...")
                    annotate_start = time.time()
                    annotated_img = self._create_annotated_image(image.copy(), predictions)
                    print(f"[Predict {os.path.basename(image_path)}] Annotation finished. (Took {time.time() - annotate_start:.2f}s)")
                    
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
                         # Decide if this should be a hard error or just proceed without annotated image
                         # raise IOError(f"Failed to save annotated image to {annotated_full_path}")
                    else:
                         print(f"[Predict {os.path.basename(image_path)}] Annotated image saved. (Took {time.time() - save_start:.2f}s)")
                         annotated_path = annotated_filename # Store only filename
                else:
                     print(f"[Predict {os.path.basename(image_path)}] No predictions found, skipping annotation.")

                total_time = time.time() - start_time
                print(f"[Predict SUCCESS {os.path.basename(image_path)}] Prediction successful. Returning results. (Total time: {total_time:.2f}s)")
                return {
                    'raw_predictions': predictions,
                    'annotated_path': annotated_path,
                    'error': None
                }
                
        except Exception as e:
            total_time = time.time() - start_time
            print(f"[Predict ERROR {os.path.basename(image_path)}] Exception occurred: {e} (Total time: {total_time:.2f}s)")
            traceback.print_exc() # Keep traceback for details
            return {
                'raw_predictions': [],
                'annotated_path': None,
                'error': str(e)
            }
        finally:
            # This block executes whether an exception occurred or not
            print(f"[Predict END {os.path.basename(image_path)}] Exiting predict method.")

    def _create_annotated_image(self, image, predictions):
        """Create annotated image with bounding boxes, labels, and masks"""
        print(f"[_Annotate START] Received {len(predictions)} predictions.")
        annotate_start_time = time.time()
        # Create a copy for drawing masks to blend later
        overlay = image.copy()
        output = image.copy()
        alpha = 0.4 # Transparency factor for masks

        print(f"[_Annotate] Starting loop over predictions...")
        for i_pred, pred in enumerate(predictions):
            print(f"[_Annotate {i_pred+1}/{len(predictions)}] Processing prediction...")
            # Extract data
            bbox = pred['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            confidence = pred['confidence']
            class_name = pred['class_name']
            mask_points = pred.get('mask') # Get mask points if they exist

            # --- Draw Mask --- 
            if mask_points:
                print(f"[_Annotate {i_pred+1}] Drawing mask...")
                mask_draw_start = time.time()
                try:
                    pts = np.array(mask_points, dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color=(0, 255, 255)) # Yellow (BGR)
                    print(f"[_Annotate {i_pred+1}] Mask drawn. (Took {time.time() - mask_draw_start:.4f}s)")
                except Exception as e_mask:
                    print(f"[_Annotate ERROR {i_pred+1}] Error drawing mask: {e_mask}")
                    traceback.print_exc() # Log error but continue if possible
            else:
                print(f"[_Annotate {i_pred+1}] No mask points found.")
            
            # --- Draw Bounding Box --- 
            print(f"[_Annotate {i_pred+1}] Drawing bounding box...")
            box_draw_start = time.time()
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
            print(f"[_Annotate {i_pred+1}] Bounding box drawn. (Took {time.time() - box_draw_start:.4f}s)")
            
            # --- Draw Label --- 
            print(f"[_Annotate {i_pred+1}] Drawing label...")
            label_draw_start = time.time()
            label = f"{class_name} ({confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # Label Background
            cv2.rectangle(output, (x1, y1-label_h-10), (x1+label_w, y1), (0, 255, 0), -1) # Green background
            # Label Text
            cv2.putText(output, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Black text
            print(f"[_Annotate {i_pred+1}] Label drawn. (Took {time.time() - label_draw_start:.4f}s)")

        print(f"[_Annotate] Finished loop. Blending images...")
        blend_start = time.time()
        # Blend the overlay with masks onto the output image with boxes/labels
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, dst=output)
        print(f"[_Annotate] Blending finished. (Took {time.time() - blend_start:.4f}s)")
        
        total_annotate_time = time.time() - annotate_start_time
        print(f"[_Annotate END] Returning annotated image. (Total time: {total_annotate_time:.2f}s)")
        return output 