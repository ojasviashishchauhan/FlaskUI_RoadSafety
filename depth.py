# depth.py - Script for Pothole Depth Estimation Testing

# Imports
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image as PILImage
import numpy as np
import cv2
import matplotlib.pyplot as plt # Keep for saving depth map visualization
import os
import traceback
import math
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions --- 
def create_boolean_mask_from_polygon(polygon_points, shape):
    """Create a boolean NumPy mask from a list of polygon points."""
    mask = np.zeros(shape, dtype=np.uint8)
    # FIX: Correct check for NumPy array
    if polygon_points is not None and polygon_points.size > 0:
    # END FIX
        try:
            # Convert points to NumPy array of int32, required by fillPoly
            pts = np.array(polygon_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1) # Fill polygon with 1
        except Exception as e:
            logger.error(f"Error creating mask from polygon: {e}")
            # Return an empty mask on error
            return np.zeros(shape, dtype=bool)
    return mask.astype(bool)

def calibrate_midas_depth(inputs):
    """Estimate metric depth (cm) based on MiDaS relative depth contrast within the mask.
    Compares percentile depth inside the mask to median depth at the boundary.
    !!! Requires tuning of SCALING_FACTOR based on real-world examples !!!
    """
    # Extract inputs
    depth_map = inputs.get('depth_map') # Expecting the full numpy depth map
    mask_polygon_points = inputs.get('mask_polygon_points') # Expecting [[x1,y1], [x2,y2], ...]
    shape = depth_map.shape if depth_map is not None else None

    if depth_map is None or mask_polygon_points is None or not mask_polygon_points.size or shape is None:
        logger.warning("Depth Calibration skipped: Missing depth_map or mask_polygon_points.")
        return 0.0

    try:
        # --- Normalize Depth Map (0-1 range) --- 
        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        if max_depth == min_depth: # Avoid division by zero for flat depth maps
            normalized_depth_map = np.zeros_like(depth_map)
            logger.warning("Depth Calibration: Depth map is flat, cannot normalize.")
        else:
            normalized_depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        # -----------------------------------------

        # 1. Create boolean mask from polygon
        inner_mask = create_boolean_mask_from_polygon(mask_polygon_points, shape)
        if not np.any(inner_mask):
            logger.warning("Depth Calibration skipped: Empty inner mask generated.")
            return 0.0

        # 2. Define boundary region (dilate mask and subtract inner)
        kernel_size = max(5, int(min(shape) * 0.015))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(inner_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary_mask = dilated_mask & ~inner_mask

        # 3. Get NORMALIZED depth values
        norm_depth_values_in_mask = normalized_depth_map[inner_mask]
        norm_depth_values_on_boundary = normalized_depth_map[boundary_mask]

        if norm_depth_values_in_mask.size == 0 or norm_depth_values_on_boundary.size == 0:
            if norm_depth_values_in_mask.size == 0:
                 logger.warning("Depth Calibration skipped: No NORMALIZED depth values found in INNER mask.")
            if norm_depth_values_on_boundary.size == 0:
                 logger.warning("Depth Calibration skipped: No NORMALIZED depth values found in BOUNDARY mask. Mask might be too close to image edge or dilation failed.")
            return 0.0

        # 4. Calculate key depth percentiles (Robust Statistics)
        # Assuming LOWER normalized value means FURTHER away (deeper)
        inner_depth_10th_percentile = np.percentile(norm_depth_values_in_mask, 10)
        boundary_depth_median = np.median(norm_depth_values_on_boundary)
        
        # 5. Calculate relative difference
        relative_difference = boundary_depth_median - inner_depth_10th_percentile
        
        # --- Constants (Require Tuning/Calibration) ---
        DEPTH_SCALING_FACTOR = 200.0 # <--- !! PLACEHOLDER !! Needs tuning!
        # --- End Constants ---

        estimated_depth_cm = 0.0
        if relative_difference > 0:
             estimated_depth_cm = relative_difference * DEPTH_SCALING_FACTOR

        logger.debug(f"Depth Calib: NormInner10pct={inner_depth_10th_percentile:.3f}, NormBoundaryMedian={boundary_depth_median:.3f}, NormRelDiff={relative_difference:.3f} -> EstDepth={estimated_depth_cm:.1f}cm (Factor={DEPTH_SCALING_FACTOR})")

        return max(0, round(estimated_depth_cm, 1))

    except Exception as e:
        logger.error(f"Error during depth calibration: {e}\n{traceback.format_exc()}")
        return 0.0

# --- Main Execution Logic ---
if __name__ == "__main__":
    logger.info("Starting Depth Estimation Script...")

    # 1. Load MiDaS Model
    processor = None
    model = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model_checkpoint = "Intel/dpt-hybrid-midas"
        processor = DPTImageProcessor.from_pretrained(model_checkpoint)
        model = DPTForDepthEstimation.from_pretrained(model_checkpoint).to(device)
        model.eval()
        logger.info(f"MiDaS model '{model_checkpoint}' loaded successfully onto {device}.")
    except Exception as e:
        logger.error(f"Error loading MiDaS model: {e}", exc_info=True)

    # 2. Load Test Image
    image_path = "/Users/ojasvi/Downloads/44.png"
    pil_image = None
    try:
        pil_image = PILImage.open(image_path).convert("RGB")
        logger.info(f"Image loaded successfully from: {image_path}")
    except FileNotFoundError:
        logger.error(f"ERROR: Image file not found at {image_path}")
    except Exception as e:
        logger.error(f"Error loading image: {e}", exc_info=True)

    # 3. Generate Relative Depth Map
    depth_map = None
    original_size = None
    if model is not None and processor is not None and pil_image is not None:
        try:
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            original_size = pil_image.size[::-1]
            prediction_interpolated = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=original_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_map = prediction_interpolated.cpu().numpy()
            logger.info(f"MiDaS inference complete. Depth map shape: {depth_map.shape}")

            # Save the depth map visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(depth_map, cmap='viridis')
            plt.title('Relative Depth Map')
            plt.colorbar(label='Relative Depth')
            plt.axis('off')
            depth_map_save_path = "depth_map_visualization.png"
            plt.savefig(depth_map_save_path)
            plt.close() # Close the figure to avoid displaying it if run non-interactively
            logger.info(f"Depth map visualization saved to: {depth_map_save_path}")
            
        except Exception as e:
            logger.error(f"Error during MiDaS inference: {e}", exc_info=True)
            depth_map = None

    # 4. Define Sample Mask & Run Calibration
    estimated_depth = 0.0
    if depth_map is not None and pil_image is not None:
        # !!! IMPORTANT: Adjust these coordinates to match a pothole in your image !!!
        img_h, img_w = depth_map.shape
        x1, y1 = int(img_w * 0.4), int(img_h * 0.6)
        x2, y2 = int(img_w * 0.6), int(img_h * 0.8)
        sample_mask_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        logger.info(f"Using sample mask polygon: {sample_mask_polygon.tolist()}")
        
        calibration_inputs = {
            'depth_map': depth_map, 
            'mask_polygon_points': sample_mask_polygon 
        }
        
        estimated_depth = calibrate_midas_depth(calibration_inputs)
        logger.info(f'Estimated Pothole Depth: {estimated_depth} cm')
        
        # Save the image with mask and depth text
        try:
            img_with_mask_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            cv2.polylines(img_with_mask_cv, [sample_mask_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(img_with_mask_cv, f"Est. Depth: {estimated_depth} cm", 
                        (sample_mask_polygon[0, 0], sample_mask_polygon[0, 1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            output_image_path = "image_with_depth_estimate.png"
            cv2.imwrite(output_image_path, img_with_mask_cv)
            logger.info(f"Image with mask and depth saved to: {output_image_path}")
        except Exception as e:
            logger.error(f"Error saving annotated image: {e}", exc_info=True)

    else:
        logger.warning("Cannot run calibration because depth map or image was not loaded/processed properly.")

    logger.info("Depth Estimation Script Finished.") 