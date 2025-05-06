import os
import datetime
from flask import render_template, redirect, url_for, flash, request, send_file, jsonify, Response, send_from_directory, stream_with_context
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from app import app, db
from app.forms import LoginForm, RegistrationForm, ImageUploadForm, VideoUploadForm
from app.models import User, Image
from app.utils import allowed_file, extract_image_metadata, extract_video_metadata, get_location_details, generate_csv_from_metadata, calculate_iou
from app.ml_predictor import MLPredictor
import uuid
import json
from PIL import Image as PILImage # Import Pillow Image
from pillow_heif import register_heif_opener # Import the HEIF opener
import subprocess # Import for running sips command
from collections import Counter, defaultdict
import calendar # For month names
from bson import ObjectId # Import ObjectId for matching
# Geopy imports
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import re # Import regex module
import requests
import math
from threading import Thread
import time
from concurrent.futures import ThreadPoolExecutor
from .chart_utils import ChartDataProcessor
from app.damage_detector import DamageDetector
import cv2 # Import cv2
import numpy as np # Import numpy for array operations
import glob
import threading
import mongoengine.errors
import traceback

# --- DeepSORT Imports ---
# from .deep_sort.deep_sort import DeepSort 
# ----------------------

# --- MiDaS Imports and Setup ---
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

midas_processor = None
midas_model = None
# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    app.logger.info("Loading MiDaS model...")
    # Use a smaller model for potentially faster inference on CPU if needed
    # model_checkpoint = "Intel/dpt-large"
    model_checkpoint = "Intel/dpt-hybrid-midas"
    midas_processor = DPTImageProcessor.from_pretrained(model_checkpoint)
    midas_model = DPTForDepthEstimation.from_pretrained(model_checkpoint).to(device)
    midas_model.eval() # Set model to evaluation mode
    app.logger.info(f"MiDaS model loaded successfully onto {device}.")
except Exception as e:
    app.logger.error(f"Error loading MiDaS model: {e}")
    app.logger.warning("Depth estimation will be disabled.")
# --- End MiDaS Setup ---

# Initialize ML predictor
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
print(f"\nInitializing MLPredictor with models directory: {models_dir}")
print(f"Available model files: {os.listdir(models_dir)}")
ml_predictor = MLPredictor(models_dir)

# Initialize damage detector with ML predictor
damage_detector = DamageDetector(ml_predictor)

# Thread pool for image processing
thread_pool = ThreadPoolExecutor(max_workers=4)

# Global lock for processing
processing_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        # Find user by username using MongoEngine
        user = User.objects(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Login failed. Please check your username and password.', 'danger')
    
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            # Create new user instance using MongoEngine
            user = User(
                username=form.username.data,
                email=form.email.data
            )
            user.set_password(form.password.data) # Hash the password
            user.save() # Save the new user document
            flash('Your account has been created! You can now log in.', 'success')
            return redirect(url_for('login'))
        except mongoengine.errors.NotUniqueError as e:
             # More specific error handling for duplicates
             app.logger.warning(f"Registration failed for {form.username.data}: Duplicate username or email.")
             flash('Registration failed: Username or Email already exists.', 'danger') 
        except Exception as e:
            # Handle potential errors like duplicate username/email
            app.logger.error(f"Registration failed for {form.username.data}: {e}")
            app.logger.error(traceback.format_exc())
            flash(f'Registration failed due to an unexpected error.', 'danger') 
            
    return render_template('register.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    # Get view preference and search term
    view = request.args.get('view', 'grid')
    search_term = request.args.get('search', '')
    
    # Build query using MongoEngine
    query = {}
    if not current_user.is_admin:
        # Use the user's actual ObjectId from the User document
        query['user_id'] = current_user.id 
    
    # Add filename search condition if term exists
    if search_term:
        query['filename__icontains'] = search_term
    
    # Fetch image documents using MongoEngine
    # Use .select_related() for potential optimization if fetching user info often
    images_qs = Image.objects(**query).order_by('-upload_time')
    
    # Prepare images list, potentially adding username for admin view
    images_list = list(images_qs) # Execute the query
    if current_user.is_admin:
        # Fetch usernames efficiently
        user_ids = [img.user_id for img in images_list]
        users = User.objects(id__in=user_ids).only('id', 'username')
        user_map = {str(user.id): user.username for user in users}
        
        # Add username to each image object (as a temporary attribute for the template)
        for img in images_list:
            img.username = user_map.get(str(img.user_id), 'Unknown User')
            
    # Pass the list of Image DOCUMENTS to the template
    return render_template('dashboard.html', 
                         images=images_list, 
                         is_admin=current_user.is_admin,
                         view=view)

def process_image_async(file_data, image_id, image_type_from_form):
    """Process image asynchronously using image_id"""
    # Get original filename for logging if needed (optional)
    try:
        temp_image = Image.objects(id=image_id).only('original_filename').first()
        original_filename_for_log = temp_image.original_filename if temp_image else str(image_id)
    except Exception:
        original_filename_for_log = str(image_id)

    # --- LOGGING: Task Start --- 
    app.logger.info(f"[PID:{os.getpid()}] process_image_async starting for: ID {image_id} (Orig: {original_filename_for_log})") 
    image = None  # Initialize image to None
    start_time = time.time() # Track processing time
    try:
        # Find the specific image entry by ID
        image = Image.objects(id=image_id).first()
        if not image:
             app.logger.error(f"Error: Could not find image entry for ID {image_id}")
             return # Exit if no entry found

        # Generate unique filename (if not already set)
        if not image.filename or not image.file_path:
            unique_filename_part = uuid.uuid4().hex
            filename_ext = os.path.splitext(image.original_filename)[1] if image.original_filename else '.png'
            image.filename = f"{unique_filename_part}{filename_ext}"
            image.file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            app.logger.info(f"Generated filename {image.filename} for ID {image_id}")

        # Save the actual image file data
        app.logger.info(f"Saving image file to {image.file_path}")
        with open(image.file_path, 'wb') as f:
            f.write(file_data)
        app.logger.info(f"Image file saved for ID {image_id}")
        
        # --- Extract and Save Metadata/Location --- 
        app.logger.info(f"Extracting metadata for {image.filename} (ID: {image.id})")
        metadata = extract_image_metadata(image.file_path)
        location = {}
        # Check for lat/lon in the primary metadata structure
        lat = metadata.get('latitude')
        lon = metadata.get('longitude')
        if lat is not None and lon is not None:
            location['latitude'] = lat
            location['longitude'] = lon
            # Get detailed location information
            try:
                location_details = get_location_details(lat, lon)
                app.logger.info(f"Location details found for ID {image.id}: {location_details}")
                # Store the entire location details dictionary
                location.update(location_details)
            except Exception as loc_err:
                app.logger.warning(f"Error getting location details for ID {image.id}: {loc_err}") # Warning, not error
                location['formatted_address'] = "Address lookup failed"
                location['city'] = "Unknown"
                location['state'] = "Unknown"
        app.logger.info(f"Assigning Metadata: {metadata} for ID {image.id}")
        app.logger.info(f"Assigning Location: {location} for ID {image.id}")
        image.metadata = metadata
        image.location = location
        # --- End Metadata/Location Extraction ---

        # Update status to processing 
        image.processing_status = 'processing'
        image.image_type = image_type_from_form 
        app.logger.info(f"Saving initial state (processing + meta/loc) for ID {image.id}")
        try:
            image.save()
            app.logger.info(f"Initial state saved for ID {image.id}")
        except Exception as save_err:
             app.logger.error(f"Error saving initial state for ID {image.id}: {save_err}")
             # Potentially mark as failed immediately? Or let subsequent steps fail?
             # Let's try to continue, prediction step will likely fail if needed
             pass 
        
        # Run ML prediction
        app.logger.info(f"Starting ML prediction for {image.filename} (ID: {image.id})")
        prediction = ml_predictor.predict(image.file_path, image.image_type)
        app.logger.info(f"ML prediction completed for {image.filename} (ID: {image.id})")
        
        # Check for prediction error from ml_predictor
        if prediction.get('error'):
             raise Exception(f"ML Prediction failed: {prediction['error']}")

        # --- Update DB with results --- 
        # Refetch to avoid potential conflicts if other processes exist (though unlikely with lock)
        image_to_update = Image.objects(id=image.id).first()
        if not image_to_update:
             app.logger.error(f"ERROR: Could not find image {image.filename} (ID: {image.id}) for final save.")
             return # Exit processing

        image_to_update.processing_status = 'completed'
        image_to_update.completion_time = datetime.datetime.now()
        
        # Log confidence score before assigning
        raw_preds = prediction.get('raw_predictions', [])
        if raw_preds:
             conf_score = max((p.get('confidence', 0.0) for p in raw_preds), default=0.0)
        else:
             conf_score = 0.0
        app.logger.info(f"Assigning confidence_score: {conf_score} for {image.filename}")
        image_to_update.confidence_score = conf_score 
        
        annotated_path = prediction.get('annotated_path')

        # --- Start MiDaS Depth Estimation & Area from Mask --- 
        midas_success = False
        if midas_model and midas_processor and raw_preds: # Only run if model loaded and detections exist
            app.logger.info(f"[PID:{os.getpid()}] Running MiDaS depth estimation for {image.filename}...")
            midas_start_time = time.time()
            try:
                # Load image with PIL for transformers
                pil_image = PILImage.open(image.file_path).convert("RGB")
                
                # Prepare input
                inputs = midas_processor(images=pil_image, return_tensors="pt").to(device)
                
                # MiDaS Inference
                with torch.no_grad():
                    outputs = midas_model(**inputs)
                    predicted_depth = outputs.predicted_depth
                
                # Interpolate depth map to original image size
                original_size = pil_image.size[::-1] # (height, width)
                prediction_interpolated = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=original_size,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                depth_map = prediction_interpolated.cpu().numpy() # Relative depth map
                midas_infer_time = time.time() - midas_start_time
                app.logger.info(f"[PID:{os.getpid()}] MiDaS inference complete for {image.filename}. Shape: {depth_map.shape} (Took {midas_infer_time:.2f}s)")

                # Calculate depth and area for each prediction using masks
                processing_errors = [] # Collect warnings/errors during metric calculation
                for pred_idx, pred in enumerate(raw_preds):
                    pred['estimated_depth_cm'] = 0 # Default
                    pred['accurate_area_pixels'] = 0 # Default
                    pred['accurate_area_m2'] = 0 # Default
                    
                    if 'mask' in pred and pred['mask'] is not None:
                        mask_polygon_points = np.array(pred['mask'], dtype=np.int32)
                        if mask_polygon_points.size == 0:
                             app.logger.warning(f"Warning: Empty mask polygon for pred {pred_idx} in {image.filename}. Skipping metrics.")
                             continue

                        # 1. Calculate Area from Mask Polygon
                        try:
                            area_pixels = cv2.contourArea(mask_polygon_points)
                            pred['accurate_area_pixels'] = area_pixels

                            # --- START SIMPLIFIED Area m2 Estimation --- 
                            if area_pixels > 0:
                                # Define a placeholder average pixels/meter value.
                                # This NEEDS TUNING based on typical camera setup and image resolution!
                                ESTIMATED_AVG_PIXELS_PER_METER = 200.0 # Example: Assumes objects are roughly scaled such that 200px = 1m
                                
                                if ESTIMATED_AVG_PIXELS_PER_METER > 0:
                                    area_m2 = area_pixels / (ESTIMATED_AVG_PIXELS_PER_METER ** 2)
                                    pred['accurate_area_m2'] = round(area_m2, 3)
                                    app.logger.debug(f"Img {image.filename}, Pred {pred_idx}: AreaPx={area_pixels:.0f}, Px/m (Avg)={ESTIMATED_AVG_PIXELS_PER_METER:.1f} -> Area={area_m2:.3f}m2")
                                else:
                                     pred['accurate_area_m2'] = 0 # Should not happen with positive constant
                            else: # if area_pixels == 0
                                pred['accurate_area_m2'] = 0 # Area is 0 if pixel area is 0
                            # --- END SIMPLIFIED Area m2 Estimation ---

                        except Exception as area_err:
                            err_msg = f"Warning: Could not calculate mask area or m2 for pred {pred_idx} in {image.filename}: {area_err}"
                            app.logger.warning(err_msg)
                            processing_errors.append(err_msg)

                        # 2. Calculate Depth from MiDaS map using Mask
                        try:
                            # --- Pass required info to calibration function --- 
                            # The calibration function now handles checks for empty masks/values internally.
                            calibration_inputs = {
                                'depth_map': depth_map, 
                                'mask_polygon_points': mask_polygon_points 
                            }
                            pred['estimated_depth_cm'] = calibrate_midas_depth(calibration_inputs)
                            # --- End Call --- 

                        except Exception as depth_err:
                             # Log the error and ensure depth is 0
                             app.logger.warning(f"Warning: Error encountered calling/during calibrate_midas_depth for pred {pred_idx} in {image.filename}: {depth_err}")
                             app.logger.error(traceback.format_exc()) # Log full traceback for debugging
                             processing_errors.append(f"Error calculating depth for pred {pred_idx}: {depth_err}")
                             pred['estimated_depth_cm'] = 0 # Set to 0 on error
                    else:
                         app.logger.warning(f"Warning: No mask found for prediction {pred_idx} in {image.filename}. Cannot calculate area/depth.")

                midas_success = True
                app.logger.info(f"[PID:{os.getpid()}] MiDaS processing finished for {image.filename}. Success: {midas_success}")
            except Exception as midas_err:
                app.logger.error(f"ERROR during MiDaS processing for {image.filename}: {midas_err}")
                app.logger.error(traceback.format_exc())
                # Keep raw_preds as they were, depth/area fields won't be added
        # --- End MiDaS --- 

        # --- Determine Primary Damage Type --- 
        primary_damage_type = 'Unknown'
        if raw_preds:
            class_names = [pred.get('class_name', 'Unknown') for pred in raw_preds]
            if class_names:
                # Count occurrences and find the most common
                from collections import Counter
                type_counts = Counter(c for c in class_names if c != 'Unknown')
                if type_counts:
                    primary_damage_type = type_counts.most_common(1)[0][0]
                elif 'Unknown' in class_names: # If only unknowns, use Unknown
                    primary_damage_type = 'Unknown'
                # If raw_preds exist but all class_names were None/empty, it stays 'Unknown'
        # --- End Determine Type --- 

        pred_results_dict = {
            'damage_detected': bool(raw_preds),
            'damage_type': primary_damage_type, # <<< ADDED KEY HERE
            'raw_predictions': raw_preds, # Include raw predictions (now potentially with depth/area)
            'annotated_path': annotated_path
        }
        # Log the prediction results dict before assigning
        app.logger.info(f"Assigning prediction_results: {pred_results_dict} for {image.filename}") 
        image_to_update.prediction_results = pred_results_dict
        
        # Extract processing time if available in prediction results
        # Assuming processing_time might be added to the prediction dict by MLPredictor
        # if 'processing_time' in prediction:
        #      image_to_update.processing_time = prediction['processing_time']
        # Or maybe calculate total time here if not passed back:
        # image_to_update.processing_time = time.time() - start_time_of_processing
        
        image_to_update.annotated_image_path = annotated_path # Save annotated path
        
        app.logger.info(f"Attempting to save final FULL results for {image.filename} (ID: {image.id})...")
        try:
            image_to_update.save()
            app.logger.info(f"Successfully saved final FULL results for {image.filename} (ID: {image.id})")
        except Exception as final_save_err:
            app.logger.error(f"ERROR saving final FULL results for {image.filename} (ID: {image.id}): {final_save_err}")
            # Log traceback for detailed error
            import traceback
            traceback.print_exc() 
            # Update status to failed if save fails
            try:
                image_to_update.processing_status = 'failed'
                image_to_update.error_message = f"Failed to save results: {final_save_err}"
                image_to_update.save()
                app.logger.info(f"Updated image ID {image.id} status to failed after save error.")
            except Exception as update_err:
                app.logger.error(f"Error updating status to failed for ID {image.id} after save error: {update_err}")
        # --- End original result processing ---
            
    except Exception as e:
        app.logger.error(f"Error in process_image_async for ID {image_id} (Orig: {original_filename_for_log}): {str(e)} (Duration: {time.time() - start_time:.2f}s)")
        app.logger.error(traceback.format_exc())
        # Update database with error status if image object exists
        if image: # Use the image object fetched at the start
            try:
                image.processing_status = 'failed'
                image.error_message = str(e)
                image.save()
                app.logger.info(f"Updated image ID {image_id} status to failed.")
            except Exception as update_err:
                app.logger.error(f"Error updating error status for ID {image_id}: {update_err}")

@app.route('/upload/image', methods=['GET', 'POST'])
@login_required
def upload_image():
    form = ImageUploadForm()
    if form.validate_on_submit():
        uploaded_count = 0
        failed_count = 0
        
        for uploaded_file in form.images.data:
            original_filename = secure_filename(uploaded_file.filename)
            current_user_id_str = current_user.get_id() # Get user ID once
            try:
                # Handle image upload
                file_data = uploaded_file.read()
                
                # Create initial database entry using MongoEngine
                initial_entry = Image(
                    original_filename=original_filename, 
                    user_id=ObjectId(current_user_id_str),
                    upload_time=datetime.datetime.now(),
                    image_type=form.image_type.data,
                    processing_status='pending' 
                )
                initial_entry.save()
                app.logger.info(f"Created initial DB entry for {original_filename} with ID: {initial_entry.id}")
                
                # Submit to thread pool OR run synchronously for Potholes
                if form.image_type.data == 'Potholes':
                    app.logger.info(f"--- Running Potholes processing SYNCHRONOUSLY for {original_filename} ---")
                    process_image_async(file_data, initial_entry.id, form.image_type.data)
                    app.logger.info(f"--- SYNCHRONOUS Potholes processing finished for {original_filename} ---")
                else:
                    future = thread_pool.submit(process_image_async, file_data, initial_entry.id, form.image_type.data)
                    # --- LOGGING: Task Submission (Moved inside else block) ---
                    app.logger.info(f"[PID:{os.getpid()}] Submitted ASYNC task for {original_filename} (User: {current_user_id_str}). Future running: {future.running()}") 
                
                uploaded_count += 1
            
            except Exception as e:
                failed_count += 1
                app.logger.error(f"Failed to initiate processing for {original_filename}: {str(e)}")
                app.logger.error(traceback.format_exc())
        
        if uploaded_count > 0:
            flash(f'{uploaded_count} file(s) uploaded and queued for processing.', 'success')
        if failed_count > 0:
            flash(f'{failed_count} file(s) failed to upload.', 'danger')
             
        return redirect(url_for('dashboard'))
    
    return render_template('upload_image.html', form=form)

# --- Video Processing Function ---
def process_video_async(file_data, video_doc_id, image_type):
    """Processes video: runs prediction on frames, generates annotated video using Ultralytics tracker."""
    logger = app.logger 
    logger.info(f"[VID_PROC START] Starting processing for video ID: {video_doc_id}, Type: {image_type}")
    video_doc = None
    cap = None
    writer = None
    annotated_video_path = None
    processed_frame_count = 0
    detection_frame_count = 0
    total_raw_detections_in_video = 0
    all_frame_detections = [] 
    # Use defaultdict for easier history tracking
    tracked_objects_history = defaultdict(lambda: {'class_name': 'Unknown', 'start_frame': -1, 'last_frame': -1, 'duration': 0, 'bboxes': [], 'confidences': []}) 

    # --- Tracking Parameters --- 
    min_confidence_threshold = 0.3 # Confidence for initial detection (used by model.track indirectly)
    tracker_config = 'botsort.yaml' # Using BoT-SORT for ReID capabilities
    # Tracker parameters are now controlled by the YAML file (e.g., botsort.yaml)
    # ---------------------------

    try:
        # --- Get the YOLO model instance from MLPredictor --- 
        yolo_model_instance = ml_predictor.models.get(image_type)
        if not yolo_model_instance:
            logger.warning(f"[VID_PROC WARN] YOLO model instance for type '{image_type}' not found in ml_predictor. Trying 'All'.")
            yolo_model_instance = ml_predictor.models.get('All') # Try fallback
            if not yolo_model_instance:
                 logger.error(f"[VID_PROC ERROR] Fallback YOLO model 'All' not found either. Cannot proceed.")
                 raise ValueError(f"YOLO model instance for type '{image_type}' or 'All' not found.")
        model_used_type = image_type if yolo_model_instance == ml_predictor.models.get(image_type) else 'All'
        logger.info(f"[VID_PROC] Using YOLO model instance for type '{model_used_type}' for tracking with {tracker_config}.")
        # --- End Model Instance --- 

        video_doc = Image.objects(id=video_doc_id).first()
        if not video_doc or video_doc.media_type != 'video':
            logger.error(f"[VID_PROC ERROR] Invalid document or not a video for ID: {video_doc_id}")
            return

        # Ensure file path exists
        if not video_doc.file_path or not os.path.exists(video_doc.file_path):
             logger.warning(f"Regenerating file path: {video_doc.file_path}")
             with open(video_doc.file_path, 'wb') as f:
                 f.write(file_data)
             logger.info("Video file saved.")
        
        input_video_path = video_doc.file_path
        
        # Extract metadata if not already present (or re-extract)
        if not video_doc.metadata or 'frame_width' not in video_doc.metadata:
             logger.info(f"Extracting video metadata for {video_doc.filename} (ID: {video_doc.id})")
             metadata = extract_video_metadata(input_video_path)
             video_doc.metadata = metadata
        else:
             # Metadata already exists, use it
             metadata = video_doc.metadata
             logger.info(f"Using existing metadata for {video_doc.filename} (ID: {video_doc.id})")

        # --- Process Location from Metadata (Similar to Image processing) --- 
        location = {} 
        lat = metadata.get('latitude')
        lon = metadata.get('longitude')
        if lat is not None and lon is not None:
             # Ensure lat/lon are floats
             try:
                 lat = float(lat)
                 lon = float(lon)
                 location['latitude'] = lat
                 location['longitude'] = lon
                 # Get detailed location information using the utility function
                 try:
                     location_details = get_location_details(lat, lon)
                     logger.info(f"Location details found for Video ID {video_doc.id}: {location_details}")
                     location.update(location_details) # Add city, state etc. to location dict
                 except Exception as loc_err:
                     logger.warning(f"Error getting location details for Video ID {video_doc.id}: {loc_err}")
                     location['formatted_address'] = "Address lookup failed"
                     location['city'] = "Unknown"
                     location['state'] = "Unknown"
             except (ValueError, TypeError) as conv_err:
                 logger.warning(f"Could not convert extracted lat/lon to float for Video ID {video_doc.id}: lat={lat}, lon={lon} - Error: {conv_err}")
        else:
             logger.info(f"Latitude/Longitude not found in metadata for Video ID {video_doc.id}. Cannot process location.")
        
        # Assign the processed location (even if empty) to the document field
        video_doc.location = location 
        # --- End Location Processing --- 

        video_doc.processing_status = 'processing'
        video_doc.image_type = image_type 
        try:
            video_doc.save() # Save processing status, metadata, AND location
            logger.info(f"Initial state (processing + meta + loc) saved for Video ID {video_doc.id}")
        except Exception as save_err:
            logger.error(f"Error saving initial state (with location) for Video ID {video_doc.id}: {save_err}")
            # If saving fails here, we should probably stop processing
            raise save_err # Re-raise the exception to be caught by the outer block

        # Check if metadata extraction failed (e.g., OpenCV couldn't open file)
        if metadata.get('error'):
            err_msg = f"Failed to get valid video metadata: {metadata.get('error', 'Unknown error')}"
            logger.error(f"[VID_PROC ERROR] {err_msg} for {video_doc.filename}")
            raise ValueError(err_msg)

        # --- Video Reading and Writing Setup --- 
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            err_msg = f"Cannot open video file: {input_video_path}"
            logger.error(f"[VID_PROC ERROR] {err_msg}")
            raise IOError(err_msg)

        # Define output path
        base, ext = os.path.splitext(video_doc.filename)
        annotated_filename = f"{base}_annotated.mp4" # Always save as mp4 for web compatibility?
        annotated_video_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)

        # Define codec and create VideoWriter object
        # Use mp4v codec for .mp4 files - TRYING avc1 instead for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        # fourcc = cv2.VideoWriter_fourcc(*'h264') # Alternative if avc1 fails
        fps = metadata.get('fps', 25.0)
        width = metadata.get('frame_width')
        height = metadata.get('frame_height')
        logger.info(f"[VID_PROC DEBUG] Creating VideoWriter: Path={annotated_video_path}, FourCC=avc1, FPS={fps}, Size=({width}x{height})")
        writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
             err_msg = f"cv2.VideoWriter failed to open. Path={annotated_video_path}, FourCC={fourcc}, FPS={fps}, Size=({width},{height})"
             logger.error(f"[VID_PROC ERROR] {err_msg}")
             raise IOError(err_msg)
        logger.info(f"[VID_PROC DEBUG] VideoWriter opened successfully.")

        logger.info(f"Starting frame processing for Video ID: {video_doc_id}")
        total_frames = metadata.get('frame_count', 0)
        frame_num = 0

        # --- Frame Processing Loop --- 
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            frame_num += 1
            # Process every Nth frame? For now, process all.
            # if frame_num % 5 != 0: 
            #    writer.write(frame) 
            #    continue 

            # Convert frame BGR to RGB (model.track expects BGR by default? Check docs, but usually BGR from cv2)
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Keep as BGR for model.track
            frame_bgr = frame

            # --- Run Ultralytics Tracking --- 
            # results = yolo_model_instance.track(source=frame_bgr, persist=True, tracker=tracker_config, conf=min_confidence_threshold, verbose=False)
            # Adding stream=True might be necessary if passing frames one by one, but let's try without first.
            # `persist=True` is key for maintaining tracks across frames.
            # Pass the frame directly. conf is applied internally.
            results = yolo_model_instance.track(source=frame_bgr, persist=True, tracker=tracker_config, verbose=False)
            # ----------------------------- 

            # --- Process Tracking Results --- 
            current_track_ids = set()
            frame_has_detections = False
            annotated_frame = frame_bgr.copy() # Start with the original frame

            if results and results[0].boxes and results[0].boxes.id is not None:
                frame_has_detections = True
                detection_frame_count += 1
                
                # Get boxes, track IDs, confidences, and classes
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                class_names_map = results[0].names # Get class names mapping from results

                # --- Store raw frame detections (optional but useful) --- 
                current_frame_raw_detections = []
                for i in range(len(track_ids)):
                    current_frame_raw_detections.append({
                        'class_name': class_names_map.get(class_ids[i], f'Class_{class_ids[i]}'),
                        'confidence': float(confidences[i]),
                        'bbox': boxes_xyxy[i].tolist(),
                        'track_id': track_ids[i] # Include track ID here too
                    })
                if current_frame_raw_detections:
                     all_frame_detections.append({
                         'frame': frame_num,
                         'detections': current_frame_raw_detections
                     })
                # --------------------------------------------------------
                
                # Plot results on the frame using Ultralytics built-in plot function
                annotated_frame = results[0].plot() 

                # --- Update Track History --- 
                for i in range(len(track_ids)):
                    track_id = track_ids[i]
                    current_track_ids.add(track_id)
                    bbox = boxes_xyxy[i]
                    conf = float(confidences[i])
                    cls_name = class_names_map.get(class_ids[i], f'Class_{class_ids[i]}')

                    track_info = tracked_objects_history[track_id] # Get or create entry
                    if track_info['start_frame'] == -1: # First time seeing this ID
                        track_info['start_frame'] = frame_num
                        track_info['class_name'] = cls_name # Assign class on first sight
                    
                    track_info['last_frame'] = frame_num
                    track_info['duration'] += 1
                    track_info['bboxes'].append(bbox.tolist()) # Store bbox history
                    track_info['confidences'].append(conf) # Store confidence history
                    # Optional: Update class name if it changes and confidence is high?
                    # if conf > 0.5 and track_info['class_name'] != cls_name:
                    #    track_info['class_name'] = cls_name
            else:
                # No tracks detected in this frame
                # annotated_frame remains the original frame copy
                pass
                
            # --- End Result Processing --- 

            # --- Draw Track History (Optional - Can make video cluttered) ---
            # Example: Draw paths for tracks seen in last N frames
            # for track_id, history in tracked_objects_history.items():
            #     if frame_num - history['last_frame'] < 10: # Only draw recent paths
            #         points = np.array([[int((b[0]+b[2])/2), int((b[1]+b[3])/2)] for b in history['bboxes'][-30:]]) # Use centroid
            #         points = points.reshape((-1, 1, 2))
            #         cv2.polylines(annotated_frame, [points], isClosed=False, color=(128, 128, 128), thickness=2)
            # ----------------------------------------------------------------

            writer.write(annotated_frame)

            # Optional: Log progress
            if frame_num % 100 == 0 and total_frames > 0:
                logger.info(f"[VID_PROC] Processed frame {frame_num}/{total_frames} for Video ID: {video_doc_id}")
            elif frame_num % 100 == 0:
                 logger.info(f"[VID_PROC] Processed frame {frame_num} for Video ID: {video_doc_id}")

        # --- End Loop --- 
        
        # --- Filter tracks by duration and finalize unique reports --- 
        final_unique_damage_reports = []
        min_track_duration_frames = 5 # Keep a minimum duration filter
        for track_id, track_data in tracked_objects_history.items():
            if track_data.get('duration', 0) >= min_track_duration_frames:
                 # Calculate average confidence for this track
                 avg_conf = sum(track_data['confidences']) / len(track_data['confidences']) if track_data['confidences'] else 0
                 final_unique_damage_reports.append({
                     'track_id': track_id,
                     'class_name': track_data['class_name'],
                     'start_frame': track_data['start_frame'],
                     'end_frame': track_data['last_frame'],
                     'duration_frames': track_data.get('duration', 0),
                     'average_confidence': round(avg_conf, 3)
                 })
        # Sort reports by start frame
        final_unique_damage_reports.sort(key=lambda x: x['start_frame'])
        # -------------------------------------------------------------
        
        logger.info(f"Finished frame processing. Found {len(final_unique_damage_reports)} unique damages meeting duration criteria ({min_track_duration_frames} frames) using Ultralytics tracker ({tracker_config}).")
        
        # --- Save Final Results --- 
        video_doc_final = Image.objects(id=video_doc_id).first()
        if not video_doc_final:
             logger.error(f"[VID_PROC ERROR] Cannot find video doc {video_doc_id} before final save.")
             return 

        video_doc_final.processing_status = 'completed'
        video_doc_final.completion_time = datetime.datetime.now()
        video_doc_final.annotated_image_path = annotated_filename # Save ANNOTATED path
        
        # --- Calculate Final Results --- 
        damage_was_detected = bool(final_unique_damage_reports)
        total_tracked_objects = len(tracked_objects_history)
        logger.info(f"[VID_PROC] Ultralytics Tracking Summary: Damage Detected (meeting criteria): {damage_was_detected} ({len(final_unique_damage_reports)} instances). Total unique tracks initiated: {total_tracked_objects}. Raw Detections recorded across frames: {len(all_frame_detections)} frames.")
        # -----------------------------

        # Update summary message and results
        video_doc_final.prediction_results = {
             'message': f'Video processing completed using Ultralytics tracker ({tracker_config}). Found {len(final_unique_damage_reports)} unique damage instance(s) meeting duration criteria ({min_track_duration_frames} frames). {total_tracked_objects} unique tracks initiated overall.',
             'unique_damage_count': len(final_unique_damage_reports), 
             'unique_damage_reports': final_unique_damage_reports, 
             'damage_detected': damage_was_detected,
             'all_frame_detections': all_frame_detections, # Raw detections per frame above threshold
             'tracking_method': f'Ultralytics ({tracker_config})', # Indicate tracking method used
             'min_duration_filter': min_track_duration_frames,
             'total_unique_tracks_initiated': total_tracked_objects
             }
        
        # Add processed detail fields
        # Calculate processing time more accurately
        if video_doc_final.upload_time:
             video_doc_final.processing_time = (datetime.datetime.now() - video_doc_final.upload_time).total_seconds()
        else:
             logger.warning(f"[VID_PROC WARN] Upload time not set for video {video_doc_id}, cannot calculate processing time.")
             video_doc_final.processing_time = None
        
        logger.info(f"Attempting final save for annotated video ID: {video_doc_id}") 
        video_doc_final.save()
        logger.info(f"[VID_PROC SUCCESS] Completed processing for video ID: {video_doc_id}")

    except Exception as e:
        logger.error(f"[VID_PROC ERROR] Error processing video ID {video_doc_id}: {e}")
        logger.error(traceback.format_exc())
        # Try to update status to failed
        try:
            video_doc_fail = Image.objects(id=video_doc_id).first()
            if video_doc_fail:
                 video_doc_fail.processing_status = 'failed'
                 video_doc_fail.error_message = str(e)
                 video_doc_fail.save()
                 logger.info(f"[VID_PROC] Updated video ID {video_doc_id} status to failed.")
            else:
                 logger.error(f"[VID_PROC ERROR] Could not find video doc {video_doc_id} to update status to failed.")
        except Exception as update_err:
            logger.error(f"[VID_PROC ERROR] Error updating error status for ID {video_doc_id}: {update_err}")
            
    finally:
        # Release resources
        if cap and cap.isOpened():
            cap.release()
        if writer and writer.isOpened():
            writer.release()
        logger.info(f"[VID_PROC END] Resources released for video ID: {video_doc_id}")

# --- End Video Processing Function --- 

@app.route('/upload/video', methods=['GET', 'POST'])
@login_required
def upload_video():
    form = VideoUploadForm() # Use the new form
    if form.validate_on_submit():
        uploaded_count = 0
        failed_count = 0
        
        for uploaded_file in form.videos.data: # Iterate through videos field
            original_filename = secure_filename(uploaded_file.filename)
            current_user_id_str = current_user.get_id()
            try:
                file_data = uploaded_file.read()
                
                # Create initial VIDEO database entry 
                initial_entry = Image(
                    original_filename=original_filename, 
                    user_id=ObjectId(current_user_id_str),
                    upload_time=datetime.datetime.now(),
                    media_type='video', # Set media type
                    processing_status='pending' 
                )
                initial_entry.save()
                app.logger.info(f"Created initial DB entry for VIDEO {original_filename} with ID: {initial_entry.id}")
                
                # Submit video processing task (using the new function)
                # Pass the selected image_type from the form
                future = thread_pool.submit(process_video_async, file_data, initial_entry.id, form.image_type.data)
                app.logger.info(f"[PID:{os.getpid()}] Submitted ASYNC task for VIDEO {original_filename} (ID: {initial_entry.id}, Type: {form.image_type.data}).") 
                uploaded_count += 1
                
            except Exception as e:
                failed_count += 1
                app.logger.error(f"Failed to initiate processing for VIDEO {original_filename}: {str(e)}")
                app.logger.error(traceback.format_exc())
        
        if uploaded_count > 0:
            flash(f'{uploaded_count} video(s) uploaded and queued for processing.', 'success')
        if failed_count > 0:
            flash(f'{failed_count} video(s) failed to upload.', 'danger')
             
        return redirect(url_for('dashboard')) # Redirect to dashboard for now
    
    # Render the video upload template
    return render_template('upload_video.html', form=form)

def is_video_file(filename):
    """Check if a file is a video based on its extension"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    return os.path.splitext(filename)[1].lower() in video_extensions

@app.route('/check_processing_status')
@login_required
def check_processing_status():
    try:
        user_id = ObjectId(current_user.get_id())
        
        # Fetch all pending/processing items
        processing_items = Image.objects(
            user_id=user_id, 
            processing_status__in=['pending', 'processing']
        ).only('id', 'original_filename', 'processing_status')

        # Fetch recently completed items (from ANY time, not just last minute)
        # We'll use this to update the UI with items that were pending before
        recently_finished_items = Image.objects(
             user_id=user_id,
             processing_status__in=['completed', 'failed'],
        ).exclude('prediction_results').limit(10)  # Limit to last 10 for efficiency

        status_list = [] 
        # Add currently processing items
        for item in processing_items:
            status_list.append({
                'id': str(item.id),
                'original_filename': item.original_filename,
                'status': item.processing_status,
                'progress': 50 if item.processing_status == 'processing' else 0,
                'media_type': getattr(item, 'media_type', 'image') # Include media type
            })

        # Add recently finished items
        finished_list = []
        for item in recently_finished_items:
             finished_list.append({
                 'id': str(item.id),
                 'original_filename': item.original_filename,
                 'status': item.processing_status,
                 'media_type': getattr(item, 'media_type', 'image'),
                 'annotated_path': item.annotated_image_path if item.media_type == 'image' else None,
                 'file_path': item.file_path if item.media_type == 'image' else item.filename, # Path for preview
                 'error_message': item.error_message,
                 'damage_detected': item.prediction_results.get('damage_detected', False) if item.processing_status == 'completed' else None,
                 'confidence_score': item.confidence_score if item.processing_status == 'completed' else None,
             })

        # Determine if all processing is complete (no items in pending/processing)
        all_processing_complete = len(status_list) == 0

        return jsonify({
            'processing_items': status_list, 
            'recently_finished_items': finished_list,
            'all_processing_complete': all_processing_complete
        })
    except Exception as e:
        print(f"Error checking processing status: {e}")
        return jsonify({'error': 'Failed to check status'}), 500

@app.route('/image/<image_id>')
@login_required
def image_details(image_id):
    try:
        # Fetch image document using MongoEngine
        image = Image.objects(id=ObjectId(image_id)).first()

        if not image:
            flash('Image not found', 'error')
            return redirect(url_for('dashboard'))

        # Check if user has permission to view this image
        if image.user_id != current_user.id and not current_user.is_admin:
            flash('You do not have permission to view this image', 'error')
            return redirect(url_for('dashboard'))

        # --- Start Edit: Add recommendation generation ---
        recommendations = None
        road_life_estimate = None
        defect_metrics = None # Initialize defect_metrics
        has_damage_for_recommendations = False

        # Only generate recommendations if damage was detected and processed
        if image.processing_status == 'completed' and image.prediction_results and image.prediction_results.get('damage_detected'):
            has_damage_for_recommendations = True
            # Extract damage type
            damage_type = image.prediction_results.get('damage_type', 'Unknown')
            if not damage_type and image.image_type:
                damage_type = image.image_type
            
            try:
                # Calculate defect metrics
                defect_metrics = calculate_defect_metrics(image)
                
                # Generate recommendations based on defect metrics and damage type
                # --- FIX: Pass only metrics dict --- 
                recommendations = generate_recommendations(defect_metrics)
                
                # Estimate road life based on defect metrics
                road_life_estimate = estimate_road_life(defect_metrics)
                # --- END FIX --- 
            except Exception as e:
                app.logger.error(f'Error generating recommendations within image_details for {image_id}: {str(e)}')
                app.logger.error(traceback.format_exc()) # Log full traceback
                flash(f'Could not generate recommendations: {str(e)}', 'warning') 
                # Continue rendering the page without recommendations
        # --- End Edit ---

        # Pass the raw Image DOCUMENT and recommendations to the template
        return render_template('image_details.html', 
                             image=image,
                             recommendations=recommendations,
                             defect_metrics=defect_metrics, # Pass metrics too, might be useful
                             road_life_estimate=road_life_estimate,
                             has_damage_for_recommendations=has_damage_for_recommendations)

    except Exception as e:
        flash(f'Error retrieving image details: {str(e)}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('dashboard'))

@app.route('/image/delete/<image_id>', methods=['POST'])
@login_required
def delete_image(image_id):
    try:
        # Find the image document using MongoEngine
        image = Image.objects(id=ObjectId(image_id)).first()

        if not image:
            flash('Image not found.', 'danger')
            return redirect(url_for('dashboard'))

        # Check permissions
        if image.user_id != current_user.id and not current_user.is_admin:
            flash('You do not have permission to delete this image.', 'danger')
            return redirect(url_for('dashboard'))

        # Attempt to delete associated files (original and annotated)
        files_to_delete = []
        if image.file_path and os.path.exists(image.file_path):
            files_to_delete.append(image.file_path)
        if image.annotated_image_path and os.path.exists(image.annotated_image_path):
             files_to_delete.append(image.annotated_image_path)

        deleted_files = True
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
                flash(f'Error deleting associated file {os.path.basename(file_path)}.', 'warning')
                deleted_files = False # Mark as partial success if file deletion fails

        # Delete the document from the database using MongoEngine
        image.delete()
        print(f"Deleted image document with ID: {image_id}")

        flash('Image deleted successfully.' if deleted_files else 'Image record deleted, but failed to remove some associated files.', 'success' if deleted_files else 'warning')
        return redirect(url_for('dashboard'))

    except Exception as e:
        print(f"Error deleting image {image_id}: {e}")
        flash('An error occurred while deleting the image.', 'danger')
        import traceback
        traceback.print_exc()
        return redirect(url_for('dashboard'))

@app.route('/image/view/<filename>')
def view_image(filename):
    if not filename:
        return "No filename specified", 400
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return f"Error retrieving image: {str(e)}", 404

@app.route('/image/view_annotated/<filename>')
def view_annotated_image(filename):
    if not filename:
        return "No filename specified", 400
    try:
        # Check if the filename contains "_annotated"
        if "_annotated" not in filename:
            filename = filename.rsplit('.', 1)
            filename = f"{filename[0]}_annotated.{filename[1]}"
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return f"Error retrieving annotated image: {str(e)}", 404

@app.route('/export/csv')
@login_required
def export_csv():
    try:
        # Build the base query using MongoEngine
        query_filters = {}
        if not current_user.is_admin:
            query_filters['user_id'] = ObjectId(current_user.get_id())
        
        # Fetch images using MongoEngine
        image_docs = Image.objects(**query_filters).order_by('-upload_time')

        # Convert documents to list of dictionaries
        images_data = [img.to_dict() for img in image_docs]
        
        # Add username if admin
        if current_user.is_admin:
            user_map = {str(user.id): user.username for user in User.objects(id__in=[img['user_id'] for img in images_data])}
            for img_data in images_data:
                img_data['username'] = user_map.get(img_data.get('user_id'), 'Unknown')

        # Define fields for CSV
        fields = [
            'id', 'username', 'original_filename', 'upload_time', 'processing_status',
            'image_type', 'confidence_score', 'processing_time', 'error_message',
            'metadata', 'location', 'prediction_results' # Include complex fields
        ]

        # Generate CSV content
        csv_content = generate_csv_from_metadata(images_data, fields)

        # Create response
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=image_export_{timestamp}.csv"}
        )

    except Exception as e:
        flash(f'Error exporting data: {str(e)}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('dashboard'))

@app.route('/map')
@login_required
def map_view():
    try:
        # Build base query using MongoEngine
        # Avoid using dot notation in the initial query
        query_filters = {
            'processing_status': 'completed',
            'location__exists': True, # Ensure location field exists
        }
        if not current_user.is_admin:
            query_filters['user_id'] = current_user.id

        # Fetch images with location data using MongoEngine
        # Select only necessary fields for performance
        all_images = Image.objects(**query_filters).only(
            'id', 'filename', 'original_filename', 'location', 'annotated_image_path',
            'prediction_results', 'confidence_score', 'image_type', 'upload_time', 'media_type'
        )
        
        # Filter images with valid location data after fetching
        images_with_location = []
        for img in all_images:
            if hasattr(img, 'location') and img.location:
                if isinstance(img.location, dict) and 'latitude' in img.location and 'longitude' in img.location:
                    if img.location['latitude'] is not None and img.location['longitude'] is not None:
                        images_with_location.append(img)

        # Check if any images were found
        if not images_with_location:
            app.logger.info(f"No images with location data found for user {current_user.id}")
            return render_template('map.html', images=[])

        return render_template('map.html', images=images_with_location)
    except mongoengine.errors.ValidationError as e:
        app.logger.error(f"MongoDB validation error in map view: {str(e)}")
        flash(f'Database validation error: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))
    except mongoengine.errors.OperationError as e:
        app.logger.error(f"MongoDB operation error in map view: {str(e)}")
        flash(f'Database operation error: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))
    except Exception as e:
        app.logger.error(f"Unexpected error in map view: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash(f'Error loading map view: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/bhuvan-proxy/<path:tile_path>')
def bhuvan_proxy(tile_path):
    try:
        z, x, y = tile_path.replace('.png', '').split('/')
        
        # Calculate bounding box for the tile
        n = 2.0 ** float(z)
        lon1 = float(x) / n * 360.0 - 180.0
        lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * float(y) / n))))
        lon2 = float(x + 1) / n * 360.0 - 180.0
        lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (float(y) + 1) / n))))
        
        bbox = f"{lon1},{lat1},{lon2},{lat2}"
        
        # Use Bhuvan's WMS service
        bhuvan_url = "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms"
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.1.1',
            'REQUEST': 'GetMap',
            'LAYERS': 'india3',
            'STYLES': '',
            'FORMAT': 'image/png',
            'TRANSPARENT': 'true',
            'HEIGHT': '256',
            'WIDTH': '256',
            'SRS': 'EPSG:4326',
            'BBOX': bbox
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) Gecko/20100101 Firefox/137.0',
            'Accept': 'image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://bhuvan.nrsc.gov.in/',
            'Origin': 'https://bhuvan.nrsc.gov.in'
        }
        
        print(f"Requesting WMS tile: {bhuvan_url} with params: {params}")  # Debug logging
        response = requests.get(bhuvan_url, params=params, headers=headers, timeout=10)
        print(f"Response status: {response.status_code}")  # Debug logging
        
        if response.status_code == 200:
            return Response(
                response.content,
                content_type=response.headers.get('content-type', 'image/png'),
                status=200
            )
        else:
            print(f"Bhuvan WMS request failed: {response.status_code} - {response.text}")  # Debug logging
            return f'Tile not found: {z}/{x}/{y}', 404
            
    except Exception as e:
        print(f"Proxy error: {str(e)}")  # Debug logging
        return str(e), 500

@app.route('/analytics')
@login_required
def analytics():
    app.logger.info("--- Entering /analytics route ---") # Add entry log
    query = {}
    user_id_for_log = 'admin' # Default for admin
    if not current_user.is_admin:
        user_id_str = current_user.get_id()
        user_id_for_log = user_id_str # Log the actual user ID
        try:
            user_id_obj = ObjectId(user_id_str)
            # Correct query for ObjectId or string representation if needed
            query['user_id'] = user_id_obj # Assume user_id is stored as ObjectId
        except Exception as e:
            app.logger.error(f"Error converting user ID {user_id_str} to ObjectId: {e}")
            # Fallback or handle error appropriately - maybe query by string if conversion fails?
            # For now, assume it's ObjectId and proceed
            query['user_id'] = user_id_obj

    try:
        # Fetch all relevant images ONCE
        app.logger.info(f"Analytics: Executing query for user '{user_id_for_log}': {query}") # Log query
        images = list(Image.objects(**query).exclude('prediction_results.all_frame_detections')) # Exclude potentially large field
        total_images = len(images)
        app.logger.info(f"Analytics: Found {total_images} images for user '{user_id_for_log}'.") # Log count found

        if not images:
            app.logger.warning(f"Analytics: No images found for user '{user_id_for_log}', rendering 'No Data'.") # Log reason for no data
            return render_template('analytics.html', has_data=False)

        # --- Initialize data structures ---
        daily_distribution_data = defaultdict(int)
        daily_activity_data = defaultdict(int)
        processing_time_by_type_data = defaultdict(list)
        confidence_by_type_data = defaultdict(list)
        damage_type_counts = defaultdict(int)
        severity_by_type_data = defaultdict(lambda: defaultdict(int))
        location_based_stats = defaultdict(lambda: {
            'total': 0,
            'damage_detected': 0,
            'confidence_sum': 0.0,
            'confidence_count': 0,
            'processing_time_sum': 0.0,
            'processing_time_count': 0,
            'severity_counts': defaultdict(int),
            'damage_types': defaultdict(int),
            'coords': []
        })
        # Use timezone-aware datetime objects
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        confidence_over_time_data = defaultdict(lambda: {'conf_sum': 0.0, 'count': 0})
        image_type_counts = defaultdict(int) # For Processing Time chart x-axis labels

        # Get date range for last 30 days (using timezone-aware dates)
        today = now_utc.date()
        thirty_days_ago = today - datetime.timedelta(days=30)
        date_range_30_days = [(today - datetime.timedelta(days=x)) for x in range(30)]

        # Initialize time-based charts with zeros
        for date_obj in date_range_30_days:
            date_str = date_obj.strftime('%Y-%m-%d')
            daily_distribution_data[date_str] = 0
            # Ensure confidence_over_time_data is initialized correctly
            confidence_over_time_data[date_str] = {'conf_sum': 0.0, 'count': 0}

        # --- FIX for Template Error: Recreate detection_stats ---
        # Initialize with default values first
        detection_stats_for_template = {
            'total': total_images,
            'damage_detected': 0, 
            'failed': 0,
            'processing': total_images, # Assume all processing initially
            'success_rate': 0,
            'detection_rate': 0
        }
        # --- END FIX ---

        # --- Single Iteration over Images ---
        completed_count = 0
        failed_count = 0
        processing_count = 0
        detected_damage_count = 0

        for img in images:
            # Status counts
            if img.processing_status == 'completed':
                completed_count += 1
            elif img.processing_status == 'failed':
                failed_count += 1
            else: # pending or processing
                processing_count += 1

            # --- Time Distribution ---
            upload_time_utc = None
            if img.upload_time:
                # Assume stored time is UTC if naive, otherwise use existing timezone
                if img.upload_time.tzinfo is None:
                    upload_time_utc = img.upload_time.replace(tzinfo=datetime.timezone.utc)
                else:
                    upload_time_utc = img.upload_time.astimezone(datetime.timezone.utc)

                # Daily Activity (Weekday)
                day_name = calendar.day_name[upload_time_utc.weekday()]
                daily_activity_data[day_name] += 1
                # Daily Distribution (Last 30 Days)
                upload_date = upload_time_utc.date()
                if upload_date >= thirty_days_ago:
                     date_str = upload_date.strftime('%Y-%m-%d')
                     # Use setdefault to ensure the key exists before incrementing
                     daily_distribution_data.setdefault(date_str, 0)
                     daily_distribution_data[date_str] += 1

            # --- Processing Time (only for completed) ---
            proc_time = None
            if img.processing_status == 'completed' and img.completion_time and upload_time_utc:
                completion_time_utc = None
                if img.completion_time.tzinfo is None:
                     completion_time_utc = img.completion_time.replace(tzinfo=datetime.timezone.utc)
                else:
                     completion_time_utc = img.completion_time.astimezone(datetime.timezone.utc)

                proc_time = (completion_time_utc - upload_time_utc).total_seconds()
                if 0 < proc_time < 7200: # Allow up to 2 hours
                     img_type = img.image_type or 'Unknown'
                     processing_time_by_type_data[img_type].append(proc_time)
                     image_type_counts[img_type]+=1
                else:
                    proc_time = None # Discard unreasonable times

            # --- Damage, Confidence, Severity (only for completed) ---
            damage_type = 'None'
            confidence = 0.0
            severity = 'Unknown'
            is_damage_detected = False

            if img.processing_status == 'completed' and img.prediction_results:
                pred_results = img.prediction_results
                is_damage_detected = pred_results.get('damage_detected', False)

                if is_damage_detected:
                    detected_damage_count += 1 # Increment overall damage count
                    # Determine damage type
                    damage_type = pred_results.get('damage_type', img.image_type or 'Unknown')
                    if not damage_type: damage_type = 'Unknown' # Ensure not None or empty

                    # Get confidence score
                    confidence = img.confidence_score or 0.0
                    # Fallback needed? Check if confidence_score is reliably populated
                    # If not, calculate from raw_predictions if necessary (omitted for brevity now)

                    # --- Calculate severity based on metrics ---
                    # This requires calling calculate_defect_metrics for every damaged image
                    # Ensure calculate_defect_metrics handles potential errors gracefully
                    try:
                         defect_metrics = calculate_defect_metrics(img)
                         severity = defect_metrics.get('severity', 'Low')
                    except Exception as metrics_err:
                         app.logger.error(f"Error calculating defect metrics for image {img.id}: {metrics_err}")
                         severity = 'Unknown' # Assign Unknown if calculation fails
                    # -----------------------------------------

                    damage_type_counts[damage_type] += 1
                    if confidence > 0: # Only include valid confidence scores
                        confidence_by_type_data[damage_type].append(confidence)
                    severity_by_type_data[damage_type][severity] += 1

                    # Confidence Over Time (only for damaged images with valid confidence and time)
                    if confidence > 0 and upload_time_utc:
                        upload_date = upload_time_utc.date()
                        if upload_date >= thirty_days_ago:
                             date_str = upload_date.strftime('%Y-%m-%d')
                             # Use setdefault for nested dict
                             conf_time_entry = confidence_over_time_data.setdefault(date_str, {'conf_sum': 0.0, 'count': 0})
                             conf_time_entry['conf_sum'] += confidence
                             conf_time_entry['count'] += 1
                # else: No damage detected for this completed image

            if not is_damage_detected and img.processing_status == 'completed':
                 damage_type_counts['None'] += 1 # Count non-damaged completed images

            # --- Location Analysis (only for completed with location) ---
            if img.processing_status == 'completed' and img.location:
                loc_info = img.location
                lat = loc_info.get('latitude')
                lon = loc_info.get('longitude')

                if lat is not None and lon is not None:
                    # Determine location key
                    city = loc_info.get('city', 'Unknown')
                    state = loc_info.get('state', 'Unknown')
                    loc_key = "Unknown Location" # Default

                    # Prioritize city, state format
                    if city and city != 'Unknown':
                        loc_key = city
                        if state and state != 'Unknown' and state != city:
                            loc_key = f"{city}, {state}"
                    elif state and state != 'Unknown':
                         loc_key = state # Use state if city unknown
                    else:
                         try:
                             # Fallback to coordinates string if no names
                             loc_key = f"Coords({round(float(lat), 3)}, {round(float(lon), 3)})"
                         except (ValueError, TypeError):
                              loc_key = "Invalid Coords" # Handle non-float coords

                    # Update stats for this location key
                    stats = location_based_stats[loc_key]
                    stats['total'] += 1
                    try:
                         stats['coords'].append({'lat': float(lat), 'lng': float(lon)})
                    except (ValueError, TypeError):
                         pass # Skip adding invalid coords

                    if is_damage_detected:
                        stats['damage_detected'] += 1
                        if confidence > 0: # Only add valid confidence
                            stats['confidence_sum'] += confidence
                            stats['confidence_count'] += 1
                        stats['severity_counts'][severity] += 1
                        stats['damage_types'][damage_type] += 1

                    if proc_time is not None: # Use valid processing time
                        stats['processing_time_sum'] += proc_time
                        stats['processing_time_count'] += 1

        # --- End Loop ---

        # --- Post-Processing Calculations ---
        app.logger.info("Analytics: Starting post-processing calculations.")

        # Processing Time by Type
        avg_time_by_type_final = {}
        total_proc_time_sum = 0
        total_proc_time_count = 0
        for img_type, times in processing_time_by_type_data.items():
            if times:
                 avg = sum(times) / len(times)
                 avg_time_by_type_final[img_type] = round(avg, 2)
                 total_proc_time_sum += sum(times)
                 total_proc_time_count += len(times)
            else:
                 avg_time_by_type_final[img_type] = 0
        # Ensure all types encountered are present
        for img_type in image_type_counts:
            avg_time_by_type_final.setdefault(img_type, 0)

        # Avg Confidence by Type
        avg_confidence_by_type_final = {}
        for d_type, conf_list in confidence_by_type_data.items():
             if conf_list:
                 avg_conf = sum(conf_list) / len(conf_list)
                 # Confidence score is 0-1, convert to percentage for display
                 avg_confidence_by_type_final[d_type] = round(avg_conf * 100, 1)
             else:
                 avg_confidence_by_type_final.setdefault(d_type, 0)
        # Ensure all damage types are present (excluding 'None')
        for d_type in damage_type_counts:
            if d_type != 'None':
                 avg_confidence_by_type_final.setdefault(d_type, 0)

        # Damage Type Distribution (ensure 'None' is included if present)
        damage_type_distribution_final = dict(damage_type_counts)

        # Confidence Over Time Trend
        confidence_trend_final = {}
        # Ensure keys exist for all dates in range and sort chronologically
        sorted_dates = sorted(date_range_30_days)
        for date_obj in sorted_dates:
            date_str = date_obj.strftime('%Y-%m-%d')
            data = confidence_over_time_data.get(date_str, {'conf_sum': 0.0, 'count': 0}) # Use get with default
            avg_conf = (data['conf_sum'] / data['count'] * 100) if data['count'] > 0 else 0
            confidence_trend_final[date_str] = round(avg_conf, 1)

        # Geographical Damage Distribution (Location vs Counts)
        geo_distribution_final = {}
        for loc_key, stats in location_based_stats.items():
            geo_distribution_final[loc_key] = {
                'total': stats['total'],
                'damaged': stats['damage_detected']
            }

        # Geographic Risk Assessment (Table Data) & Top Risk Areas
        risk_assessment_entries = []
        for loc_key, stats in location_based_stats.items():
            if stats['total'] == 0: continue # Skip locations with no images somehow

            avg_conf = (stats['confidence_sum'] / stats['confidence_count'] * 100) if stats['confidence_count'] > 0 else 0
            avg_proc = (stats['processing_time_sum'] / stats['processing_time_count']) if stats['processing_time_count'] > 0 else 0
            damage_rate = (stats['damage_detected'] / stats['total'] * 100) if stats['total'] > 0 else 0

            # Calculate Risk Score (Example logic)
            risk_score = 0
            risk_score += damage_rate / 5 # Higher damage rate increases score
            risk_score += stats['severity_counts'].get('High', 0) * 5 # High severity adds more
            risk_score += stats['severity_counts'].get('Medium', 0) * 2 # Medium severity adds less
            risk_score += avg_conf / 20 # Higher avg confidence might indicate denser/clearer damage

            # Determine Risk Level String
            risk_level = "Low"
            if risk_score > 15: risk_level = "High" # Adjusted threshold
            elif risk_score > 5: risk_level = "Medium" # Adjusted threshold

            risk_assessment_entries.append({
                'location': loc_key,
                'total_inspections': stats['total'],
                'damage_detected': stats['damage_detected'],
                'damage_rate': round(damage_rate, 1), # Add damage rate
                'avg_confidence': round(avg_conf, 1),
                'avg_processing_time': round(avg_proc, 2),
                'risk_level': risk_level,
                'risk_score': risk_score # Keep score for sorting
            })

        # Sort by risk score (descending) for the table and top areas
        risk_assessment_entries.sort(key=lambda x: x['risk_score'], reverse=True)
        # Final table data (without score)
        geographic_risk_assessment_final = [{k: v for k, v in entry.items() if k != 'risk_score'} for entry in risk_assessment_entries]
        # Top 5 Risk Areas data (simple format for chart)
        top_risk_areas_final = [{'name': entry['location'], 'score': round(entry['risk_score'],1)} for entry in risk_assessment_entries[:5]]

        # System Reliability (Success Rate based on completed vs total)
        system_reliability = (completed_count / total_images * 100) if total_images > 0 else 100

        # Overall Stats Card Data
        stats_summary = {
            'total_inspections': total_images,
            'damage_detection_rate': (detected_damage_count / completed_count * 100) if completed_count > 0 else 0,
            'avg_processing_time': (total_proc_time_sum / total_proc_time_count) if total_proc_time_count > 0 else 0,
            'system_reliability': system_reliability
        }

        # --- FIX for Template Error: Re-calculate detection_stats ---
        # Update the initialized dict with calculated values
        detection_stats_for_template['damage_detected'] = detected_damage_count
        detection_stats_for_template['failed'] = failed_count
        detection_stats_for_template['processing'] = processing_count # Actual pending/processing
        detection_stats_for_template['success_rate'] = stats_summary['system_reliability']
        detection_stats_for_template['detection_rate'] = stats_summary['damage_detection_rate']
        # total remains as total_images from initialization
        # --- END FIX ---

        # Daily Activity (Weekday) - ensure correct order
        sorted_daily_activity = {day: daily_activity_data.get(day, 0) for day in calendar.day_name}

        # Daily Distribution (Last 30 days) - ensure correct date order
        sorted_daily_distribution = {date_obj.strftime('%Y-%m-%d'): daily_distribution_data.get(date_obj.strftime('%Y-%m-%d'), 0) for date_obj in sorted_dates}

        app.logger.info(f"Analytics: Data processing complete. Passing {len(geographic_risk_assessment_final)} locations to template.")

        # Pass the FINAL data structures to the template
        return render_template('analytics.html',
            has_data=True,
            stats_summary=stats_summary, # Keep the new summary
            detection_stats=detection_stats_for_template, # Pass the reconstructed old stats for template compatibility
            # Chart Data
            daily_distribution=sorted_daily_distribution,
            daily_activity=sorted_daily_activity,
            processing_time_by_type=avg_time_by_type_final,
            avg_confidence_by_type=avg_confidence_by_type_final,
            damage_type_distribution=damage_type_distribution_final,
            confidence_trend=confidence_trend_final,
            geographical_distribution=geo_distribution_final,
            top_risk_areas=top_risk_areas_final,
            # Table Data
            geographic_risk_assessment=geographic_risk_assessment_final,
            # Make sure all necessary variables are passed
        )

    except Exception as e:
        app.logger.error(f"Error in analytics route: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc()) # Log the full traceback
        return render_template('analytics.html',
                             has_data=False,
                             error_message=f"An error occurred while generating analytics: {str(e)}", # Pass error message
                             show_error=True,
                             # --- FIX: Pass default data in except block --- 
                             stats_summary={}, 
                             # Pass the initialized empty stats dict here too
                             detection_stats={'total': 0, 'damage_detected': 0, 'failed': 0, 'processing': 0, 'success_rate': 0, 'detection_rate': 0}, 
                             daily_distribution={}, 
                             daily_activity={}, 
                             processing_time_by_type={}, 
                             avg_confidence_by_type={}, 
                             # --- FIX: Add missing default --- 
                             avg_time_by_type={}, 
                             # --- END FIX --- 
                             damage_type_distribution={}, 
                             confidence_trend={}, 
                             geographical_distribution={}, 
                             top_risk_areas=[], 
                             geographic_risk_assessment=[],
                             # ADD DEFAULTS FOR MISSING VARIABLES
                             processing_times={'avg_time': 0, 'median_time': 0, 'min_time': 0, 'max_time': 0}, # Added default
                             damage_types=[] # Added default
                             # --- END FIX --- 
                             )

@app.route('/delete_stuck_images', methods=['POST'])
@login_required
def delete_stuck_images():
    if not current_user.is_admin:
        flash('Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # Define cutoff time (e.g., images stuck in 'processing' for more than 1 hour)
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        
        # Find stuck images using MongoEngine
        stuck_images = Image.objects(
            processing_status='processing',
            upload_time__lt=cutoff_time # Check upload_time as creation marker
        )
        
        deleted_count = 0
        file_errors = 0
        
        for image in stuck_images:
            try:
                # Attempt to delete associated files
                if image.file_path and os.path.exists(image.file_path):
                    try:
                        os.remove(image.file_path)
                    except OSError as e:
                        print(f"Error deleting file {image.file_path}: {e}")
                        file_errors += 1
                
                # Delete the image document
                image.delete()
                deleted_count += 1
            except Exception as inner_e:
                print(f"Error deleting stuck image {image.id}: {inner_e}")
        
        message = f"Attempted to delete {deleted_count} stuck image(s)." 
        if file_errors > 0:
             message += f" Failed to delete {file_errors} associated file(s)."
        flash(message, 'success' if file_errors == 0 else 'warning')
        
    except Exception as e:
        flash(f'An error occurred while deleting stuck images: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()
        
    return redirect(url_for('dashboard'))

@app.route('/image/stop_and_delete/<image_id>', methods=['POST'])
@login_required
def stop_and_delete_image(image_id):
    if not current_user.is_admin:
        flash('Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))

    try:
        # Find the image using MongoEngine
        image = Image.objects(id=ObjectId(image_id)).first()
        
        if not image:
            flash('Image not found.', 'danger')
            return jsonify({'success': False, 'message': 'Image not found'}), 404

        # Check if it's actually processing (optional, could delete regardless)
        if image.processing_status not in ['pending', 'processing']:
             flash(f'Image {image.original_filename or image.id} is not currently processing.', 'info')
             # Optionally, still allow deletion or return an error
             # return jsonify({'success': False, 'message': 'Image not processing'}), 400

        print(f"Attempting to stop and delete image {image.id}...")
        # NOTE: Stopping the actual thread in thread_pool is complex and 
        # often not practical/safe. The simplest approach is to just delete 
        # the record and files. The background thread will eventually error out 
        # when it tries to save results to the non-existent DB record.
        
        # Attempt to delete associated files
        files_to_delete = []
        if image.file_path and os.path.exists(image.file_path):
            files_to_delete.append(image.file_path)
        if image.annotated_image_path and os.path.exists(image.annotated_image_path):
             files_to_delete.append(image.annotated_image_path)

        deleted_files = True
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path} for image {image.id}: {e}")
                deleted_files = False

        # Delete the document from the database
        image.delete()
        print(f"Deleted image document {image.id}")
        
        message = 'Processing stopped and image deleted.' if deleted_files else 'Processing stopped, image record deleted, but failed to remove files.'
        flash(message, 'success' if deleted_files else 'warning')
        return jsonify({'success': True, 'message': message})

    except Exception as e:
        print(f"Error stopping/deleting image {image_id}: {e}")
        flash('An error occurred during stop/delete.', 'danger')
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error during deletion'}), 500

# Add a new route to display debug images
@app.route('/debug')
@login_required
def debug_view():
    # Only allow admin users to view debug info
    if not current_user.is_admin:
        flash('You need admin privileges to view debug information.', 'warning')
        return redirect(url_for('dashboard'))
    
    # Get the debug directory
    debug_dir = os.path.join(app.root_path, '..', 'static', 'debug')
    
    # Try to read the latest debug info
    debug_info = {}
    try:
        with open(os.path.join(debug_dir, "debug_info.txt"), "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    debug_info[key.strip()] = value.strip()
    except:
        flash('No debug information found.', 'warning')
        return redirect(url_for('dashboard'))
    
    # Get the latest debug filename
    latest_debug = debug_info.get('Latest debug images')
    if not latest_debug:
        flash('No debug images found.', 'warning')
        return redirect(url_for('dashboard'))
    
    # Get all image files for this debug session
    debug_images = sorted(glob.glob(os.path.join(debug_dir, f"{latest_debug}_*.jpg")))
    
    # Convert to relative paths for display
    debug_image_paths = ['/'.join(img.split(os.sep)[-3:]) for img in debug_images]
    
    # Group images by type
    image_groups = {
        'Original & Preprocessing': [],
        'Thresholding': [],
        'Contours': [],
        'Bounding Boxes': [],
        'Masks': [],
        'Other': []
    }
    
    for img_path in debug_image_paths:
        img_name = os.path.basename(img_path)
        if any(x in img_name for x in ['gray', 'blurred', 'original']):
            image_groups['Original & Preprocessing'].append(img_path)
        elif any(x in img_name for x in ['thresh', 'binary', 'otsu']):
            image_groups['Thresholding'].append(img_path)
        elif any(x in img_name for x in ['contour']):
            image_groups['Contours'].append(img_path)
        elif 'bbox' in img_name:
            image_groups['Bounding Boxes'].append(img_path)
        elif 'mask' in img_name:
            image_groups['Masks'].append(img_path)
        else:
            image_groups['Other'].append(img_path)
    
    return render_template(
        'debug.html',
        debug_info=debug_info,
        image_groups=image_groups,
        title='Debug Images'
    )

@app.route('/city_analytics')
@login_required
def city_analytics():
    query = {}
    if not current_user.is_admin:
        user_id_str = current_user.get_id()
        try:
            user_id_obj = ObjectId(user_id_str)
            query['$or'] = [{'user_id': user_id_str}, {'user_id': user_id_obj}]
        except Exception:
            query['user_id'] = user_id_str

    try:
        # Get all images with location data
        images = Image.objects(query).filter(location__exists=True, processing_status='completed')
        
        app.logger.info(f"Found {len(images)} images with location data for city analytics")
        
        if not images:
            return render_template('city_analytics.html', has_data=False)
        
        # City-based damage analysis
        city_stats = {}
        
        for img in images:
            # Skip if no valid location data
            if not img.location or not (img.location.get('city') or img.location.get('latitude')):
                continue
                
            # Use structured location data
            city = img.location.get('city')
            if not city or city == 'Unknown':
                # Try extracting from formatted address
                address = img.location.get('formatted_address', img.location.get('address', ''))
                if address:
                    city = extract_city_from_address(address)
                
                # Use coordinates as fallback identifier if no city name available
                if not city and img.location.get('latitude') and img.location.get('longitude'):
                    lat = round(float(img.location.get('latitude')), 2)
                    lng = round(float(img.location.get('longitude')), 2)
                    city = f"Location ({lat}, {lng})"
                elif not city:
                    city = "Unknown Location"
            
            # Get damage type from prediction results or image type
            damage_type = 'None'
            if img.prediction_results and img.prediction_results.get('damage_detected'):
                damage_type = img.prediction_results.get('damage_type', 'Unknown')
            elif hasattr(img, 'image_type') and img.image_type:
                damage_type = img.image_type
            
            # Initialize city entry if it doesn't exist
            if city not in city_stats:
                city_stats[city] = {
                    'total': 0,
                    'damage_types': defaultdict(int),
                    'total_with_damage': 0,
                    'lat': img.location.get('latitude'),
                    'lng': img.location.get('longitude'),
                    'state': img.location.get('state', 'Unknown'),
                    'country': img.location.get('country', 'Unknown')
                }
            
            # Update city statistics
            city_stats[city]['total'] += 1
            
            if damage_type != 'None':
                city_stats[city]['damage_types'][damage_type] += 1
                city_stats[city]['total_with_damage'] += 1
        
        # Debug output
        app.logger.info(f"Generated stats for {len(city_stats)} cities")
        
        # Convert defaultdicts to regular dicts for JSON serialization
        for city, stats in city_stats.items():
            stats['damage_types'] = dict(stats['damage_types'])
            
            # Calculate damage percentage
            if stats['total'] > 0:
                stats['damage_percentage'] = round((stats['total_with_damage'] / stats['total']) * 100, 1)
            else:
                stats['damage_percentage'] = 0
        
        # Sort cities by total images
        sorted_city_stats = dict(sorted(city_stats.items(), key=lambda x: x[1]['total'], reverse=True))
        
        # --- Calculate Overall Summary Stats --- 
        total_cities = len(sorted_city_stats)
        total_images_all = sum(stats['total'] for stats in sorted_city_stats.values())
        total_damage_all = sum(stats['total_with_damage'] for stats in sorted_city_stats.values())
        overall_damage_perc = (total_damage_all / total_images_all * 100) if total_images_all > 0 else 0
        
        highest_damage_city = {'name': 'N/A', 'perc': 0}
        if sorted_city_stats:
            city_with_highest = max(sorted_city_stats.items(), key=lambda x: x[1]['damage_percentage'])
            highest_damage_city = {'name': city_with_highest[0], 'perc': city_with_highest[1]['damage_percentage']}
            
        overall_summary = {
            'total_cities': total_cities,
            'avg_damage_perc': round(overall_damage_perc, 1),
            'highest_damage_city_name': highest_damage_city['name'],
            'highest_damage_city_perc': highest_damage_city['perc']
        }
        # --- End Calculate Overall Summary --- 
        
        app.logger.info(f"Rendering city analytics with data for {len(sorted_city_stats)} cities")
        
        # Check if request is AJAX
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        # Render template with different options based on request type
        template = render_template('city_analytics.html', 
                            has_data=True,
                            city_stats=sorted_city_stats,
                            overall_summary=overall_summary)
                            
        return template
    
    except Exception as e:
        app.logger.error(f"Error in city_analytics route: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return render_template('city_analytics.html', 
                             has_data=False, 
                             error_message=str(e),
                             show_error=True)

def extract_city_from_address(address):
    """Extract city name from a full address string"""
    if not address:
        return None
        
    # Try to extract city using different patterns
    
    # Pattern 1: Look for common city identifiers
    address_parts = address.split(',')
    for part in address_parts:
        # Clean up the part
        part = part.strip()
        
        # Skip very short parts or parts that are likely postal codes
        if len(part) < 3 or part.isdigit() or any(char.isdigit() for char in part):
            continue
            
        # Skip common non-city parts
        if any(skip in part.lower() for skip in ['state', 'country', 'province', 'district']):
            continue
            
        # This is likely a city
        return part
    
    # Pattern 2: If no city was found, use the first substantial part
    for part in address_parts:
        part = part.strip()
        if len(part) >= 3:
            return part
    
    # If all else fails, return the whole address
    return address

@app.route('/purge_all_data', methods=['POST'])
@login_required
def purge_all_data():
    """Purge all data from the system - admin use only"""
    if not current_user.is_admin:
        flash('Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # Get all images from the database
        all_images = Image.objects.all()
        total_count = all_images.count()
        
        if total_count == 0:
            flash('No data to purge.', 'info')
            return redirect(url_for('dashboard'))
        
        deleted_count = 0
        failed_files = 0
        
        # Delete all files and database records
        for image in all_images:
            try:
                # Try to delete the original file
                if image.file_path and os.path.exists(image.file_path):
                    try:
                        os.remove(image.file_path)
                    except OSError:
                        failed_files += 1
                
                # Try to delete the annotated file if it exists
                if image.annotated_image_path:
                    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                                 image.annotated_image_path)
                    if os.path.exists(annotated_path):
                        try:
                            os.remove(annotated_path)
                        except OSError:
                            failed_files += 1
                
                # Delete the database record
                image.delete()
                deleted_count += 1
                
            except Exception as e:
                app.logger.error(f"Error deleting image {image.id}: {str(e)}")
                continue
        
        # Provide feedback to the user
        if failed_files > 0:
            flash(f'Purge completed. Deleted {deleted_count} records from database, but failed to delete {failed_files} files.', 'warning')
        else:
            flash(f'Purge completed. Deleted all {deleted_count} records and associated files.', 'success')
            
    except Exception as e:
        app.logger.error(f"Error during purge operation: {str(e)}")
        traceback.print_exc()
        flash(f'An error occurred during purge: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))

@app.route('/state_analytics') # Renamed route
@login_required
def state_analytics(): # Renamed function
    """Provide analytics based on statewide damage data, aggregated by state.""" # Updated docstring
    query = {}
    if not current_user.is_admin:
        user_id_str = current_user.get_id()
        try:
            user_id_obj = ObjectId(user_id_str)
            query['$or'] = [{'user_id': user_id_str}, {'user_id': user_id_obj}]
        except Exception:
            query['user_id'] = user_id_str

    try:
        # Fetch completed images with valid location (including state)
        images = Image.objects(query).filter(
            location__exists=True,
            location__state__exists=True, # Ensure state field exists
            processing_status='completed'
        ).exclude('prediction_results.all_frame_detections') # Exclude large field

        app.logger.info(f"State Analytics: Found {len(images)} completed images with location for potential analysis.")

        # --- Group images by state and calculate stats ---
        state_stats = defaultdict(lambda: {
            'total_images': 0,
            'damage_detected_count': 0,
            'damage_types': defaultdict(int),
            'confidence_sum': 0.0,
            'confidence_count': 0,
            'severity_counts': defaultdict(int),
            'total_area_m2': 0.0,
            'total_depth_cm': 0.0,
            'depth_count': 0,
            'coords': [] # Store first coords for map marker
        })
        
        valid_images_for_stats = 0
        processed_states = set()

        for img in images:
            # Ensure state is valid and not empty/None/'Unknown'
            state = img.location.get('state')
            if not state or state == 'Unknown':
                continue # Skip images without a valid state

            # --- Ensure lat/lng are valid floats before adding coords ---
            lat = img.location.get('latitude')
            lng = img.location.get('longitude')
            valid_coords = False
            try:
                if lat is not None and lng is not None:
                    lat = float(lat)
                    lng = float(lng)
                    valid_coords = True
            except (ValueError, TypeError):
                 pass # Invalid coords, don't add them
            # --- End coord validation ---

            stats = state_stats[state]
            stats['total_images'] += 1
            valid_images_for_stats += 1
            processed_states.add(state)

            # Store coords of first valid image for map marker
            if valid_coords and not stats['coords']:
                 stats['coords'] = [{'lat': lat, 'lng': lng}]

            # Check for detected damage
            damage_detected = False
            primary_damage_type = 'None'
            confidence = 0.0
            severity = 'Low' # Default if no damage or metrics fail

            if img.prediction_results and img.prediction_results.get('damage_detected'):
                damage_detected = True
                stats['damage_detected_count'] += 1
                
                # Use calculated metrics for severity, area, depth
                try:
                    metrics = calculate_defect_metrics(img)
                    primary_damage_type = metrics.get('damage_type', 'Unknown')
                    severity = metrics.get('severity', 'Low')
                    area_m2 = metrics.get('area_m2', 0.0)
                    depth_cm = metrics.get('depth_cm', 0.0)
                    
                    stats['total_area_m2'] += area_m2
                    if depth_cm > 0:
                        stats['total_depth_cm'] += depth_cm
                        stats['depth_count'] += 1
                        
                except Exception as metrics_err:
                    app.logger.error(f"Error calculating metrics for {img.id} in state {state}: {metrics_err}")
                    # Fallback using basic info if metrics fail
                    primary_damage_type = img.prediction_results.get('damage_type', img.image_type or 'Unknown')
                    severity = 'Unknown' # Indicate metrics failed

                stats['damage_types'][primary_damage_type if primary_damage_type else 'Unknown'] += 1
                stats['severity_counts'][severity] += 1

                # Add confidence score if available
                conf_score = img.confidence_score or 0.0
                if conf_score > 0:
                    stats['confidence_sum'] += conf_score
                    stats['confidence_count'] += 1
            else:
                 # Still count non-damaged images towards totals
                 stats['damage_types']['None'] += 1 # Count non-damaged
                 # Severity is considered 'Low' implicitly if no damage

        if valid_images_for_stats == 0:
             app.logger.warning("State Analytics: No images with valid state information found.")
             return render_template('state_analytics.html', has_data=False)

        # --- Post-process state stats ---
        final_state_stats = {}
        for state, stats in state_stats.items():
            damage_rate = (stats['damage_detected_count'] / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
            avg_confidence = (stats['confidence_sum'] / stats['confidence_count'] * 100) if stats['confidence_count'] > 0 else 0 # Percentage
            avg_depth = (stats['total_depth_cm'] / stats['depth_count']) if stats['depth_count'] > 0 else 0.0

            final_state_stats[state] = {
                'total_images': stats['total_images'],
                'damage_detected_count': stats['damage_detected_count'],
                'damage_rate': round(damage_rate, 1),
                'avg_confidence': round(avg_confidence, 1),
                'damage_types': dict(stats['damage_types']), # Convert defaultdict
                'severity_counts': dict(stats['severity_counts']), # Convert defaultdict
                'total_area_m2': round(stats['total_area_m2'], 2),
                'avg_depth_cm': round(avg_depth, 1),
                'coords': stats['coords'][0] if stats['coords'] else None # Pass first coord dict or None
            }

        # Sort stats by state name
        sorted_final_stats = dict(sorted(final_state_stats.items()))
        
        # Calculate overall summary
        total_states_analyzed = len(sorted_final_stats)
        overall_total_images = sum(s['total_images'] for s in sorted_final_stats.values())
        overall_damage_count = sum(s['damage_detected_count'] for s in sorted_final_stats.values())
        overall_damage_rate = (overall_damage_count / overall_total_images * 100) if overall_total_images > 0 else 0
        
        state_with_highest_rate = {'name': 'N/A', 'rate': 0}
        if sorted_final_stats:
            highest = max(sorted_final_stats.items(), key=lambda item: item[1]['damage_rate'])
            state_with_highest_rate = {'name': highest[0], 'rate': highest[1]['damage_rate']}

        overall_summary = {
             'total_states': total_states_analyzed,
             'total_images': overall_total_images,
             'overall_damage_rate': round(overall_damage_rate, 1),
             'highest_damage_state': state_with_highest_rate['name'],
             'highest_damage_rate': state_with_highest_rate['rate'],
        }

        app.logger.info(f"State Analytics: Prepared data for {len(sorted_final_stats)} states.")
        
        return render_template('state_analytics.html',
                              has_data=True,
                              state_stats=sorted_final_stats, # Pass processed stats
                              overall_summary=overall_summary # Pass overall summary
                             )

    except Exception as e:
        app.logger.error(f"Error in state analytics route: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return render_template('state_analytics.html', # Keep template name
                             has_data=False,
                             error_message="An error occurred while generating state analytics.", # Updated message
                             show_error=True)

def calculate_cluster_center(points):
    """Calculate the center point of a cluster"""
    if not points:
        return {'lat': 0, 'lng': 0}
    
    sum_lat = sum(p['lat'] for p in points)
    sum_lng = sum(p['lng'] for p in points)
    return {
        'lat': sum_lat / len(points),
        'lng': sum_lng / len(points)
    }

def calculate_distance(point1, point2):
    """Calculate distance between two geographic points in kilometers using Haversine formula"""
    # Earth radius in kilometers
    R = 6371.0
    
    lat1 = math.radians(point1['lat'])
    lon1 = math.radians(point1['lng'])
    lat2 = math.radians(point2['lat'])
    lon2 = math.radians(point2['lng'])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

def cluster_by_distance(points, max_distance_km):
    """Cluster points based on geographic distance"""
    if not points:
        return []
    
    # Initialize clusters
    clusters = []
    unclustered = points.copy()
    
    while unclustered:
        # Create a new cluster with the first unclustered point
        current_cluster = [unclustered.pop(0)]
        
        # Find all points within max_distance of any point in current cluster
        i = 0
        while i < len(unclustered):
            # Check distance to any point in the current cluster
            for cluster_point in current_cluster:
                dist = calculate_distance(cluster_point, unclustered[i])
                if dist <= max_distance_km:
                    # Add to cluster and remove from unclustered
                    current_cluster.append(unclustered.pop(i))
                    # Reset index since we removed an item
                    i -= 1
                    break
            i += 1
        
        # Add completed cluster to clusters
        clusters.append(current_cluster)
    
    return clusters

def calculate_cluster_severity(cluster, cluster_distance_km):
    """Calculate severity of a cluster based on number of points and distance"""
    if not cluster:
        return "Low"
    
    # Calculate density (points per square km)
    # Approximate area as circle with radius = cluster_distance_km
    area = math.pi * (cluster_distance_km ** 2)
    density = len(cluster) / area
    
    # Average confidence
    avg_confidence = sum((p.confidence_score or 0.0) for p in cluster) / len(cluster)
    
    # Determine severity based on density and confidence
    if density > 0.5 and avg_confidence > 0.7:
        return "High"
    elif density > 0.2 or avg_confidence > 0.6:
        return "Medium"
    else:
        return "Low"

def calculate_defect_metrics(image):
    """Calculate area, depth, severity, and other metrics for defects.
    Prioritizes using accurate metrics (area_m2, depth_cm) if available in raw_predictions.
    """
    # Initialize with defaults
    defect_metrics = {
        'area_m2': 0.0,
        'depth_cm': 0.0, 
        'volume_m3': 0.0,
        'severity': 'Low',
        'confidence': image.confidence_score or 0.0,
        'damage_type': 'Unknown',
        'calculation_method': 'None' # Track how metrics were calculated
    }
    
    try:
        if not image.prediction_results or image.processing_status != 'completed':
            app.logger.warning(f"Skipping metrics calculation for {image.id}: No results or not completed.")
            defect_metrics['calculation_method'] = 'Skipped (No Data)'
            return defect_metrics # Return default if no results

        raw_predictions = image.prediction_results.get('raw_predictions', [])
        if not raw_predictions:
            app.logger.info(f"No raw predictions found for {image.id}. Using defaults.")
            defect_metrics['calculation_method'] = 'Skipped (No Detections)'
            return defect_metrics # Return default if no detections

        # --- Aggregate Metrics from Raw Predictions ---
        total_accurate_area_m2 = 0.0
        depths_cm = []
        damage_types = set()
        has_accurate_metrics = False
        calculation_method = 'Fallback Estimation' # Default if specific keys aren't found

        # Check if the first prediction has the enhanced keys we expect
        first_pred = raw_predictions[0]
        if 'accurate_area_m2' in first_pred and 'estimated_depth_cm' in first_pred:
            has_accurate_metrics = True
            calculation_method = 'MiDaS Enhanced'
            app.logger.info(f"Detected accurate metrics (MiDaS Enhanced) for image {image.id}")
        else:
            app.logger.warning(f"Accurate metrics (accurate_area_m2, estimated_depth_cm) not found in raw_predictions for image {image.id}. Will use Fallback Estimation.")

        for pred in raw_predictions:
            # Always collect damage types
            damage_types.add(pred.get('class_name', 'Unknown'))

            if has_accurate_metrics:
                area_m2 = pred.get('accurate_area_m2', 0.0)
                depth_cm = pred.get('estimated_depth_cm', 0.0)
                total_accurate_area_m2 += area_m2 if area_m2 is not None else 0.0
                if depth_cm is not None and depth_cm > 0: # Only consider valid depths
                    depths_cm.append(depth_cm)
            # else: Fallback logic will be applied later if needed

        # --- Assign Aggregated Values (Prioritize Accurate) ---
        if has_accurate_metrics:
            defect_metrics['area_m2'] = round(total_accurate_area_m2, 3)
            if depths_cm:
                # Use average of valid depths found
                defect_metrics['depth_cm'] = round(np.mean(depths_cm), 1)
            # else: depth_cm remains 0.0
            defect_metrics['calculation_method'] = calculation_method
            
        # --- Fallback Estimation (If Accurate Metrics Weren't Available) ---
        elif raw_predictions: # Only run fallback if there were predictions but no accurate keys
            calculation_method = 'Bounding Box Estimation'
            app.logger.warning(f"Executing Fallback Estimation for image {image.id}")
            metadata = image.metadata or {}
            img_width = metadata.get('width', 0)
            img_height = metadata.get('height', 0)

            if img_width > 0 and img_height > 0:
                estimated_road_width_m = 4.0 
                estimated_pixels_per_meter = img_width / estimated_road_width_m
                
                total_bbox_area_pixels = 0
                max_confidence = 0.0
                for pred in raw_predictions:
                    bbox = pred.get('bbox', [0, 0, 0, 0])
                    if len(bbox) == 4:
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        total_bbox_area_pixels += width * height
                    max_confidence = max(max_confidence, pred.get('confidence', 0.0))
                
                if estimated_pixels_per_meter > 0:
                    area_m2_est = total_bbox_area_pixels / (estimated_pixels_per_meter ** 2)
                    defect_metrics['area_m2'] = round(area_m2_est, 3)
                # else: area remains 0.0

                # Crude depth estimation based on confidence and estimated area
                primary_damage_type_fallback = image.prediction_results.get('damage_type', 'Unknown') 
                if not primary_damage_type_fallback or primary_damage_type_fallback == 'Unknown':
                    if damage_types: primary_damage_type_fallback = list(damage_types)[0] # Use first detected type
                
                depth_cm_est = 0
                confidence = max_confidence # Use max confidence from bboxes
                if 'Pothole' in primary_damage_type_fallback:
                    depth_cm_est = (confidence * 5) + (defect_metrics['area_m2'] * 2) 
                    depth_cm_est = min(depth_cm_est, 15) # Cap fallback depth at 15cm
                elif 'Crack' in primary_damage_type_fallback:
                    depth_cm_est = confidence * 3
                    depth_cm_est = min(depth_cm_est, 5) # Cap crack depth est
                defect_metrics['depth_cm'] = round(depth_cm_est, 1)
                defect_metrics['calculation_method'] = calculation_method
            else:
                 app.logger.warning(f"Cannot perform Fallback Estimation for {image.id}: Missing image dimensions.")
                 defect_metrics['calculation_method'] = 'Skipped (Missing Dims)'
        # --- End Fallback --- 

        # --- Determine Primary Damage Type (Consolidated) ---
        primary_damage_type = 'Unknown'
        if len(damage_types) == 1:
            primary_damage_type = list(damage_types)[0]
        elif len(damage_types) > 1:
            # Use the type determined during processing if available, else 'Mixed'
            primary_damage_type = image.prediction_results.get('damage_type', 'Mixed')
        else:
            # Fallback if damage_types set is empty for some reason
            primary_damage_type = image.prediction_results.get('damage_type', image.image_type or 'Unknown')
        # Ensure it's not None or empty string
        defect_metrics['damage_type'] = primary_damage_type if primary_damage_type else 'Unknown'

        # --- Calculate Severity (Based on final area/depth) ---
        if defect_metrics['area_m2'] > 1.5 or defect_metrics['depth_cm'] > 7.0:
            defect_metrics['severity'] = 'High'
        elif defect_metrics['area_m2'] > 0.5 or defect_metrics['depth_cm'] > 3.0:
            defect_metrics['severity'] = 'Medium'
        else:
            defect_metrics['severity'] = 'Low'
        # Adjust severity for Alligator cracking - always at least Medium if area > 0.1
        if 'Alligator' in defect_metrics['damage_type'] and defect_metrics['area_m2'] > 0.1 and defect_metrics['severity'] == 'Low':
            defect_metrics['severity'] = 'Medium'
             
        # --- Calculate Volume (Only for Potholes, using final area/depth) ---
        if 'Pothole' in defect_metrics['damage_type'] and defect_metrics['area_m2'] > 0 and defect_metrics['depth_cm'] > 0:
            # Volume (m3) = Area (m2) * Depth (m)
            volume_m3 = defect_metrics['area_m2'] * (defect_metrics['depth_cm'] / 100.0)
            defect_metrics['volume_m3'] = round(volume_m3, 4) # Increase precision for volume
        # else: volume remains 0.0

        app.logger.info(f"Calculated defect metrics for {image.id} (Method: {defect_metrics['calculation_method']}): {defect_metrics}")
        return defect_metrics
    
    except Exception as e:
        app.logger.error(f"Error calculating defect metrics for {image.id}: {str(e)}\n{traceback.format_exc()}")
        # Return default metrics with error noted
        defect_metrics['error'] = str(e) 
        defect_metrics['calculation_method'] = 'Error'
        return defect_metrics

def generate_recommendations(metrics):
    """Generate repair recommendations based on calculated metrics, including material suggestions and structured cost estimation."""
    recommendations = []
    
    # Extract metrics for clarity
    damage_type = metrics.get('damage_type', 'Unknown')
    severity = metrics.get('severity', 'Low')
    area_m2 = metrics.get('area_m2', 0.0)
    depth_cm = metrics.get('depth_cm', 0.0)
    volume_m3 = metrics.get('volume_m3', 0.0) # Volume specifically for Potholes
    calculation_method = metrics.get('calculation_method', 'Unknown')

    # --- Costing Parameters (Move to config or constants file later?) ---
    # Material Costs (Example values in ₹ per unit)
    COST_HOT_MIX_ASPHALT_PER_M3 = 12000 
    COST_COLD_MIX_ASPHALT_PER_M3 = 9000 
    COST_CRACK_SEALANT_PER_METER = 150 # Assume cracks are treated linearly for cost
    COST_SURFACE_TREATMENT_PER_M2 = 500 # For sealing larger areas
    COST_MILLING_PER_M2 = 300
    
    # Labor Costs (Example values in ₹)
    LABOR_BASE_RATE_PER_JOB = 3000 # Base mobilization/setup
    LABOR_RATE_PER_M2_PATCHING = 800
    LABOR_RATE_PER_METER_CRACK_SEAL = 50
    LABOR_RATE_PER_M2_OVERLAY = 600
    
    # Equipment Costs (Example - simplified)
    EQUIPMENT_BASE_COST = 2000 
    EQUIPMENT_HEAVY_DUTY_ADDON = 5000 # For milling/deep patching
    
    # Minimum total cost
    MINIMUM_REPAIR_COST = 5000 
    # --------------------------------------------------------------------

    # --- Helper Function for Cost Estimation (More Structured) ---
    def estimate_structured_cost(repair_type, area, depth=None, volume=None, crack_length_est=None):
        material_cost = 0
        labor_cost = LABOR_BASE_RATE_PER_JOB
        equipment_cost = EQUIPMENT_BASE_COST
        cost_details = [] # To explain the calculation

        if repair_type == 'Full Depth Patch':
            if volume and volume > 0:
                 material_cost = volume * COST_HOT_MIX_ASPHALT_PER_M3
                 cost_details.append(f"Material (Hot Mix): {volume:.3f} m³ * ₹{COST_HOT_MIX_ASPHALT_PER_M3}/m³ = ₹{material_cost:.0f}")
            else: # Fallback if volume is zero but area/depth exist
                 est_volume = area * (depth / 100.0) if depth else area * 0.05 # Assume 5cm depth if unknown
                 material_cost = est_volume * COST_HOT_MIX_ASPHALT_PER_M3
                 cost_details.append(f"Material (Hot Mix, Est.): {est_volume:.3f} m³ * ₹{COST_HOT_MIX_ASPHALT_PER_M3}/m³ = ₹{material_cost:.0f}")
            labor_cost += area * LABOR_RATE_PER_M2_PATCHING
            equipment_cost += EQUIPMENT_HEAVY_DUTY_ADDON # Needs heavy equipment
            cost_details.append(f"Labor (Base + Patching): ₹{LABOR_BASE_RATE_PER_JOB} + {area:.2f} m² * ₹{LABOR_RATE_PER_M2_PATCHING}/m² = ₹{labor_cost:.0f}")
            cost_details.append(f"Equipment (Base + Heavy): ₹{EQUIPMENT_BASE_COST} + ₹{EQUIPMENT_HEAVY_DUTY_ADDON} = ₹{equipment_cost:.0f}")
        
        elif repair_type == 'Partial Depth Patch (Hot Mix)':
             # Assume partial depth uses roughly 60% of full volume?
             est_volume = (volume * 0.6) if volume and volume > 0 else (area * (depth / 100.0) * 0.6 if depth else area * 0.03)
             material_cost = est_volume * COST_HOT_MIX_ASPHALT_PER_M3
             cost_details.append(f"Material (Hot Mix, Partial): {est_volume:.3f} m³ * ₹{COST_HOT_MIX_ASPHALT_PER_M3}/m³ = ₹{material_cost:.0f}")
             labor_cost += area * LABOR_RATE_PER_M2_PATCHING * 0.8 # Slightly less labor?
             equipment_cost += EQUIPMENT_HEAVY_DUTY_ADDON * 0.5 # Less heavy equip? 
             cost_details.append(f"Labor (Base + Patching): ₹{LABOR_BASE_RATE_PER_JOB} + {area:.2f} m² * ₹{LABOR_RATE_PER_M2_PATCHING*0.8}/m² = ₹{labor_cost:.0f}")
             cost_details.append(f"Equipment (Base + Medium): ₹{EQUIPMENT_BASE_COST} + ₹{EQUIPMENT_HEAVY_DUTY_ADDON*0.5} = ₹{equipment_cost:.0f}")

        elif repair_type == 'Surface Patch (Cold Mix)':
             # Assume cold mix uses less volume
             est_volume = (volume * 0.4) if volume and volume > 0 else (area * (depth / 100.0) * 0.4 if depth else area * 0.02)
             material_cost = est_volume * COST_COLD_MIX_ASPHALT_PER_M3
             cost_details.append(f"Material (Cold Mix): {est_volume:.3f} m³ * ₹{COST_COLD_MIX_ASPHALT_PER_M3}/m³ = ₹{material_cost:.0f}")
             labor_cost += area * LABOR_RATE_PER_M2_PATCHING * 0.6 # Less labor
             # Basic equipment cost only
             cost_details.append(f"Labor (Base + Patching): ₹{LABOR_BASE_RATE_PER_JOB} + {area:.2f} m² * ₹{LABOR_RATE_PER_M2_PATCHING*0.6}/m² = ₹{labor_cost:.0f}")
             cost_details.append(f"Equipment (Base): ₹{equipment_cost:.0f}")

        elif repair_type == 'Mill and Overlay':
            # Cost based on area
            material_cost = area * COST_SURFACE_TREATMENT_PER_M2 # Simplified overlay material cost
            milling_cost = area * COST_MILLING_PER_M2
            cost_details.append(f"Material (Overlay): {area:.2f} m² * ₹{COST_SURFACE_TREATMENT_PER_M2}/m² = ₹{material_cost:.0f}")
            cost_details.append(f"Milling: {area:.2f} m² * ₹{COST_MILLING_PER_M2}/m² = ₹{milling_cost:.0f}")
            labor_cost += area * LABOR_RATE_PER_M2_OVERLAY
            equipment_cost += EQUIPMENT_HEAVY_DUTY_ADDON # Milling needs heavy equipment
            cost_details.append(f"Labor (Base + Overlay): ₹{LABOR_BASE_RATE_PER_JOB} + {area:.2f} m² * ₹{LABOR_RATE_PER_M2_OVERLAY}/m² = ₹{labor_cost:.0f}")
            cost_details.append(f"Equipment (Base + Heavy): ₹{EQUIPMENT_BASE_COST} + ₹{EQUIPMENT_HEAVY_DUTY_ADDON} = ₹{equipment_cost:.0f}")
            material_cost += milling_cost # Add milling to material/prep cost
        
        elif repair_type == 'Rout and Seal' or repair_type == 'Crack Sealing':
            # Estimate crack length based on area (highly approximate)
            # Assume average crack width (e.g., 1cm = 0.01m), length = area / width
            est_length_m = crack_length_est if crack_length_est else (area / 0.01 if area > 0 else 10) # Default 10m if area 0?
            est_length_m = max(1.0, est_length_m) # Min 1m length
            material_cost = est_length_m * COST_CRACK_SEALANT_PER_METER
            cost_details.append(f"Material (Sealant): {est_length_m:.1f} m * ₹{COST_CRACK_SEALANT_PER_METER}/m = ₹{material_cost:.0f}")
            labor_cost += est_length_m * LABOR_RATE_PER_METER_CRACK_SEAL
            # Basic equipment for sealing
            cost_details.append(f"Labor (Base + Sealing): ₹{LABOR_BASE_RATE_PER_JOB} + {est_length_m:.1f} m * ₹{LABOR_RATE_PER_METER_CRACK_SEAL}/m = ₹{labor_cost:.0f}")
            cost_details.append(f"Equipment (Base): ₹{equipment_cost:.0f}")
            if repair_type == 'Rout and Seal':
                 equipment_cost += 500 # Add small cost for routing equipment
                 cost_details.append(f"Equipment Addon (Routing): ₹500")

        elif repair_type == 'Surface Treatment Seal': # For larger alligator areas not needing milling
             material_cost = area * COST_SURFACE_TREATMENT_PER_M2
             cost_details.append(f"Material (Surface Seal): {area:.2f} m² * ₹{COST_SURFACE_TREATMENT_PER_M2}/m² = ₹{material_cost:.0f}")
             labor_cost += area * LABOR_RATE_PER_M2_OVERLAY * 0.5 # Less labor than full overlay
             # Basic equipment
             cost_details.append(f"Labor (Base + Sealing): ₹{LABOR_BASE_RATE_PER_JOB} + {area:.2f} m² * ₹{LABOR_RATE_PER_M2_OVERLAY*0.5}/m² = ₹{labor_cost:.0f}")
             cost_details.append(f"Equipment (Base): ₹{equipment_cost:.0f}")

        else: # Default/Unknown
            cost_details.append("Requires On-Site Assessment for Accurate Costing.")
            return 'Requires Assessment', cost_details

        total_cost = material_cost + labor_cost + equipment_cost
        # Apply minimum cost
        final_cost = max(MINIMUM_REPAIR_COST, total_cost)
        cost_details.append(f"Subtotal: ₹{total_cost:.0f}")
        if final_cost == MINIMUM_REPAIR_COST and total_cost < MINIMUM_REPAIR_COST:
            cost_details.append(f"Applied Minimum Job Cost: ₹{MINIMUM_REPAIR_COST:.0f}")
            
        return f'₹{int(final_cost)}', cost_details # Return integer value for display
    # --- End Cost Helper --- 

    # --- Recommendation Logic --- 
    if 'Pothole' in damage_type:
        # Define materials based on repair type
        material_high = "High-Performance Hot Mix Asphalt (HMA) - e.g., Polymer Modified Binder (PMB)"
        material_medium = "Standard Hot Mix Asphalt (HMA) - e.g., VG30/VG40 Grade Bitumen"
        material_low = "Cold Mix Asphalt Patching Compound"
        
        if severity == 'High' or depth_cm > 7 or area_m2 > 2:
            repair_type = 'Full Depth Patch'
            cost, cost_breakdown = estimate_structured_cost(repair_type, area_m2, depth_cm, volume=volume_m3)
            recommendations.append({
                'title': repair_type,
                'description': f'Critical Pothole ({area_m2:.2f} m², {depth_cm:.1f} cm depth, {volume_m3:.4f} m³ volume). Requires removing damaged pavement down to a stable base and replacing with new, compacted asphalt layers.',
                'urgency': 'High',
                'materials': material_high,
                'estimated_cost': cost,
                'cost_breakdown': cost_breakdown
            })
        elif severity == 'Medium' or depth_cm > 3 or area_m2 > 0.5: # Lowered threshold slightly
            repair_type = 'Partial Depth Patch (Hot Mix)'
            cost, cost_breakdown = estimate_structured_cost(repair_type, area_m2, depth_cm, volume=volume_m3)
            recommendations.append({
                'title': repair_type,
                'description': f'Significant Pothole ({area_m2:.2f} m², {depth_cm:.1f} cm depth, {volume_m3:.4f} m³ volume). Requires cleaning, squaring edges, applying tack coat, and filling with compacted hot mix asphalt.',
                'urgency': 'Medium',
                'materials': material_medium,
                'estimated_cost': cost,
                'cost_breakdown': cost_breakdown
            })
        else:
            repair_type = 'Surface Patch (Cold Mix)'
            cost, cost_breakdown = estimate_structured_cost(repair_type, area_m2, depth_cm, volume=volume_m3)
            recommendations.append({
                'title': repair_type,
                'description': f'Minor Pothole ({area_m2:.2f} m², {depth_cm:.1f} cm depth). Suitable for temporary repair or low-traffic areas. Clean, fill with cold mix compound, and compact.',
                'urgency': 'Low',
                'materials': material_low,
                'estimated_cost': cost,
                'cost_breakdown': cost_breakdown
            })
    
    elif 'Crack' in damage_type:
        material_sealant = "Flexible Polymer-Modified Bitumen Sealant or Rubberized Asphalt Sealant"
        material_overlay = "Standard Hot Mix Asphalt (HMA) Overlay (e.g., 40-50mm thickness)"
        material_patch_seal = "Slurry Seal or Microsurfacing Treatment"
        
        # Estimate crack length for costing (very rough)
        estimated_crack_length_m = area_m2 / 0.01 if area_m2 > 0.01 else None # Assume 1cm avg width
        
        if 'Alligator' in damage_type:
            if severity == 'High' or area_m2 > 5: # Needs structural repair
                repair_type = 'Mill and Overlay'
                cost, cost_breakdown = estimate_structured_cost(repair_type, area_m2)
                recommendations.append({
                    'title': f'{repair_type} (Alligator Cracking)',
                    'description': f'Extensive alligator cracking ({area_m2:.2f} m²), indicating base fatigue. Requires milling the distressed layer(s) and applying a new structural asphalt overlay.',
                    'urgency': 'High',
                    'materials': f"Milling Debris Removal + {material_overlay}",
                    'estimated_cost': cost,
                    'cost_breakdown': cost_breakdown
                })
            else: # Medium severity alligator cracking
                repair_type = 'Surface Treatment Seal'
                cost, cost_breakdown = estimate_structured_cost(repair_type, area_m2)
                recommendations.append({
                    'title': f'{repair_type} (Alligator Cracking)',
                    'description': f'Moderate alligator cracking ({area_m2:.2f} m²). Apply a structural surface treatment (e.g., Slurry Seal, Microsurfacing) to seal cracks and prevent water ingress.',
                    'urgency': 'Medium',
                    'materials': material_patch_seal,
                    'estimated_cost': cost,
                    'cost_breakdown': cost_breakdown
                })
        elif any(x in damage_type for x in ['Longitudinal', 'Transverse']):
            # Use depth_cm as proxy for crack width/severity if area is small
            is_wide_crack = depth_cm > 1.5 # Treat depth > 1.5cm as a wider crack needing routing
            if severity == 'High' or is_wide_crack:
                 repair_type = 'Rout and Seal'
                 cost, cost_breakdown = estimate_structured_cost(repair_type, area_m2, crack_length_est=estimated_crack_length_m)
                 recommendations.append({
                    'title': f'{repair_type} (Linear Cracks)',
                    'description': f'Significant or wide linear cracking ({area_m2:.2f} m², Depth Proxy: {depth_cm:.1f} cm). Cracks should be routed/widened to create a reservoir, cleaned, and filled with flexible sealant.',
                    'urgency': 'High' if severity == 'High' else 'Medium', # Wide cracks are medium urgency
                    'materials': material_sealant,
                    'estimated_cost': cost,
                    'cost_breakdown': cost_breakdown
                })
            else: # Minor/narrow cracks
                 repair_type = 'Crack Sealing'
                 cost, cost_breakdown = estimate_structured_cost(repair_type, area_m2, crack_length_est=estimated_crack_length_m)
                 recommendations.append({
                    'title': f'{repair_type} (Linear Cracks)',
                    'description': f'Minor linear cracking ({area_m2:.2f} m², Depth Proxy: {depth_cm:.1f} cm). Clean cracks thoroughly and apply sealant to prevent water intrusion.',
                    'urgency': 'Low',
                    'materials': material_sealant,
                    'estimated_cost': cost,
                    'cost_breakdown': cost_breakdown
                })
        else: # Other crack types
             repair_type = 'Crack Sealing (General)'
             cost, cost_breakdown = estimate_structured_cost(repair_type, area_m2, crack_length_est=estimated_crack_length_m)
             recommendations.append({
                'title': repair_type,
                'description': f'General cracking detected ({area_m2:.2f} m²). Requires cleaning and sealing with appropriate sealant.',
                'urgency': severity,
                'materials': material_sealant,
                'estimated_cost': cost,
                'cost_breakdown': cost_breakdown
            })
    
    else: # Unknown or other damage types
        cost, cost_breakdown = estimate_structured_cost('Unknown', area_m2)
        recommendations.append({
            'title': 'On-Site Inspection Required',
            'description': f'Damage type uncertain or not categorized ({damage_type}, Area: {area_m2:.2f} m², Severity: {severity}). A detailed on-site inspection by a qualified engineer is required to determine the exact nature and extent of the damage and recommend the appropriate repair method.',
            'urgency': severity,
            'materials': 'To be determined after inspection',
            'estimated_cost': cost, # Still show 'Requires Assessment'
            'cost_breakdown': cost_breakdown
        })
    
    # Add general monitoring recommendation if not High urgency
    high_urgency_present = any(rec['urgency'] == 'High' for rec in recommendations)
    if not high_urgency_present:
        recommendations.append({
            'title': 'Preventative Maintenance Monitoring',
            'description': f'Current defect severity is {severity}. Recommend regular monitoring (e.g., annually or semi-annually) as part of a preventative maintenance schedule to track any deterioration and address issues proactively.',
            'urgency': 'Info',
            'materials': 'N/A',
            'estimated_cost': 'N/A',
            'cost_breakdown': ['Part of standard road maintenance plan.']
        })

    # Add note about calculation method if not ideal
    if calculation_method != 'MiDaS Enhanced' and calculation_method != 'None':
         recommendations.append({
             'title': 'Note on Metrics Accuracy',
             'description': f'The area/depth metrics used for these recommendations were based on {calculation_method}. Accuracy may be limited compared to MiDaS-enhanced or on-site measurements. Consider field verification for high-cost repairs.',
             'urgency': 'Info',
             'materials': 'N/A',
             'estimated_cost': 'N/A',
             'cost_breakdown': []
         })
            
    return recommendations

def estimate_road_life(metrics):
    """Estimate remaining road life based on damage metrics"""
    # Base estimate - typical asphalt road lasts 15-20 years
    base_life_years = 20
    
    # Reduction factors
    # --- Get damage_type from metrics --- 
    damage_type = metrics.get('damage_type', 'Unknown')
    # --- End Get --- 
    severity = metrics.get('severity', 'Low')
    area_m2 = metrics.get('area_m2', 0)
    depth_cm = metrics.get('depth_cm', 0)
    
    # Calculate reduction based on severity
    if severity == 'High':
        reduction_factor = 0.7  # Reduce life by 70%
    elif severity == 'Medium':
        reduction_factor = 0.4  # Reduce life by 40%
    else:
        reduction_factor = 0.2  # Reduce life by 20%
    
    # Additional reduction based on defect type
    if 'Pothole' in damage_type:
        defect_factor = 0.8  # Potholes reduce life more significantly
    elif 'Alligator' in damage_type:
        defect_factor = 0.7  # Alligator cracking indicates structural issues
    elif any(x in damage_type for x in ['Longitudinal', 'Transverse']):
        defect_factor = 0.5  # Linear cracks are somewhat less severe
    else:
        defect_factor = 0.3  # Unknown defect types
    
    # Calculate remaining life
    remaining_years = base_life_years * (1 - (reduction_factor * defect_factor))
    
    # Adjust based on area and depth (larger/deeper defects reduce life more)
    area_factor = min(area_m2 / 10, 0.5)  # Cap at 50% reduction for area
    depth_factor = min(depth_cm / 15, 0.5)  # Cap at 50% reduction for depth
    
    remaining_years = remaining_years * (1 - area_factor) * (1 - depth_factor)
    
    # Final estimate
    return {
        'years_remaining': max(round(remaining_years, 1), 0.1),  # Rename key
        'condition': get_condition_from_years(remaining_years),
        'notes': generate_life_estimate_notes(damage_type, metrics),
        'condition_color': get_condition_color(remaining_years) # Add color for badge
    }

def get_condition_from_years(years):
    """Get road condition description from remaining years"""
    if years < 2:
        return "Critical - Requires Immediate Attention"
    elif years < 5:
        return "Poor - Major Repairs Needed"
    elif years < 10:
        return "Fair - Regular Maintenance Required"
    elif years < 15:
        return "Good - Minor Maintenance Needed"
    else:
        return "Excellent - Regular Monitoring Only"

# --- Add helper for badge color ---
def get_condition_color(years):
    if years < 2:
        return "danger"
    elif years < 5:
        return "danger"
    elif years < 10:
        return "warning"
    elif years < 15:
        return "success"
    else:
        return "primary"
# --- End Add --- 

def generate_life_estimate_notes(damage_type, metrics):
    """Generate additional notes for the road life estimate"""
    notes = []
    
    severity = metrics.get('severity', 'Low')
    
    # Add notes based on damage type
    if 'Pothole' in damage_type:
        notes.append("Potholes indicate advanced pavement deterioration and may suggest underlying structural issues.")
        if metrics.get('depth_cm', 0) > 5:
            notes.append("Deep potholes suggest base or subgrade problems that may require full-depth repair.")
    
    elif 'Alligator' in damage_type:
        notes.append("Alligator cracking indicates fatigue damage from repeated loading, often a structural problem.")
        notes.append("Without treatment, water infiltration will accelerate deterioration.")
    
    elif 'Longitudinal' in damage_type:
        notes.append("Longitudinal cracks may indicate joint failures or improper construction.")
    
    elif 'Transverse' in damage_type:
        notes.append("Transverse cracks typically result from thermal stresses or shrinkage.")
    
    # Add severity-based notes
    if severity == 'High':
        notes.append("Immediate action recommended to prevent rapid deterioration and safety hazards.")
    elif severity == 'Medium':
        notes.append("Scheduled repair within 1-2 years is recommended to prevent further deterioration.")
    else:
        notes.append("Monitoring recommended; repair during next scheduled maintenance cycle.")
    
    # Join the list into a single string
    return " ".join(notes) 

# --- MiDaS Helper Functions ---
def create_boolean_mask_from_polygon(polygon_points, shape):
    """Create a boolean NumPy mask from a list of polygon points."""
    mask = np.zeros(shape, dtype=np.uint8)
    # --- FIX: Correct check for NumPy array --- 
    if polygon_points is not None and polygon_points.size > 0:
    # --- END FIX ---
        try:
            # Convert points to NumPy array of int32, required by fillPoly
            pts = np.array(polygon_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1) # Fill polygon with 1
        except Exception as e:
            app.logger.error(f"Error creating mask from polygon: {e}")
            # Return an empty mask on error
            return np.zeros(shape, dtype=bool)
    return mask.astype(bool)

# --- Start: Updated Depth Calibration Function (from depth.py) ---
def calibrate_midas_depth(inputs):
    """Estimate metric depth (cm) based on MiDaS relative depth contrast within the mask.
    Compares percentile depth inside the mask to median depth at the boundary.
    !!! Requires tuning of SCALING_FACTOR based on real-world examples !!!
    """
    # Extract inputs
    depth_map = inputs.get('depth_map') # Expecting the full numpy depth map
    mask_polygon_points = inputs.get('mask_polygon_points') # Expecting [[x1,y1], [x2,y2], ...]
    shape = depth_map.shape if depth_map is not None else None

    # Use app logger within the Flask app context
    if depth_map is None or mask_polygon_points is None or not mask_polygon_points.size or shape is None:
        app.logger.warning("Depth Calibration skipped: Missing depth_map or mask_polygon_points.")
        return 0.0

    try:
        # --- Normalize Depth Map (0-1 range) --- 
        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        if max_depth == min_depth: # Avoid division by zero for flat depth maps
            normalized_depth_map = np.zeros_like(depth_map)
            app.logger.warning("Depth Calibration: Depth map is flat, cannot normalize.")
        else:
            normalized_depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        # -----------------------------------------

        # 1. Create boolean mask from polygon
        inner_mask = create_boolean_mask_from_polygon(mask_polygon_points, shape)
        if not np.any(inner_mask):
            app.logger.warning("Depth Calibration skipped: Empty inner mask generated.")
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
                 app.logger.warning("Depth Calibration skipped: No NORMALIZED depth values found in INNER mask.")
            if norm_depth_values_on_boundary.size == 0:
                 app.logger.warning("Depth Calibration skipped: No NORMALIZED depth values found in BOUNDARY mask. Mask might be too close to image edge or dilation failed.")
            return 0.0

        # 4. Calculate key depth percentiles (Robust Statistics)
        # Assuming LOWER normalized value means FURTHER away (deeper)
        inner_depth_10th_percentile = np.percentile(norm_depth_values_in_mask, 10)
        boundary_depth_median = np.median(norm_depth_values_on_boundary)
        
        # 5. Calculate relative difference
        relative_difference = boundary_depth_median - inner_depth_10th_percentile
        
        # --- Constants (Require Tuning/Calibration) ---
        # Use the factor tuned in depth.py - double check this value!
        DEPTH_SCALING_FACTOR = 200.0 # <--- !! PLACEHOLDER !! Needs tuning!
        # --- End Constants ---

        estimated_depth_cm = 0.0
        if relative_difference > 0:
             estimated_depth_cm = relative_difference * DEPTH_SCALING_FACTOR

        app.logger.debug(f"Depth Calib: NormInner10pct={inner_depth_10th_percentile:.3f}, NormBoundaryMedian={boundary_depth_median:.3f}, NormRelDiff={relative_difference:.3f} -> EstDepth={estimated_depth_cm:.1f}cm (Factor={DEPTH_SCALING_FACTOR})")

        return max(0, round(estimated_depth_cm, 1))

    except Exception as e:
        # Use app logger and include traceback
        app.logger.error(f"Error during depth calibration: {e}\n{traceback.format_exc()}")
        return 0.0
# --- End: Updated Depth Calibration Function ---

# Helper function (example - if needed elsewhere)# def some_helper():
#    pass
# Make sure this is the last function before the end of the file or other non-route code

@app.route('/road_segments')
def road_segments():
    """Display damages grouped into road segments based on geographic proximity."""
    query = {}
    if not current_user.is_admin:
        user_id_str = current_user.get_id()
        try:
            # Ensure ObjectId is imported from bson
            from bson import ObjectId
            user_id_obj = ObjectId(user_id_str)
            query['user_id'] = user_id_obj
        except Exception:
            query['user_id'] = user_id_str # Fallback if ID isn't ObjectId? Adjust as needed.

    try:
        # Ensure request is imported from flask
        from flask import request, render_template
        # Ensure math is imported
        import math
        # Ensure app logger is available (imported from app)
        from app import app
        # Ensure Image model is imported
        from app.models import Image
        # Ensure Counter, defaultdict are imported
        from collections import Counter, defaultdict
        # Ensure helper functions are defined above or imported
        # (cluster_by_distance, calculate_cluster_center, calculate_cluster_severity, calculate_defect_metrics)

        # Get clustering distance from request, default to 10km
        cluster_distance_km = float(request.args.get('distance', 10)) # Allow user to set distance

        # Fetch completed images with damage and valid location data
        images = Image.objects(query).filter(
            location__exists=True,
            processing_status='completed',
            prediction_results__damage_detected=True
        ).exclude('prediction_results.all_frame_detections') # Exclude large field

        app.logger.info(f"Road Segments: Found {len(images)} images with damage for analysis.")

        if not images:
            return render_template('road_segments.html', has_data=False, cluster_distance=cluster_distance_km)

        # Prepare data points for clustering
        points = []
        image_map = {} # Store full image object for later retrieval
        for img in images:
            if img.location and 'latitude' in img.location and 'longitude' in img.location:
                lat = img.location.get('latitude')
                lng = img.location.get('longitude')
                if lat is not None and lng is not None:
                    try:
                        point_data = {
                            'id': str(img.id),
                            'lat': float(lat),
                            'lng': float(lng),
                        }
                        points.append(point_data)
                        image_map[str(img.id)] = img
                    except (ValueError, TypeError):
                        app.logger.warning(f"Skipping image {img.id} due to invalid coordinates: lat={lat}, lng={lng}")
                        continue

        if not points:
             app.logger.warning("Road Segments: No valid points found after filtering coordinates.")
             return render_template('road_segments.html', has_data=False, cluster_distance=cluster_distance_km)

        # Perform clustering
        clusters_of_points = cluster_by_distance(points, cluster_distance_km)
        app.logger.info(f"Road Segments: Generated {len(clusters_of_points)} clusters using {cluster_distance_km}km distance.")

        # Analyze clusters and prepare for template
        segment_data = []
        for i, cluster_points in enumerate(clusters_of_points):
            if not cluster_points: continue

            segment_images = [image_map[p['id']] for p in cluster_points if p['id'] in image_map]
            if not segment_images: continue

            segment_center = calculate_cluster_center(cluster_points)
            segment_severity = calculate_cluster_severity(segment_images, cluster_distance_km)

            # Determine Segment Name
            cluster_name = f"Segment #{i+1}" # Default
            road_names_in_cluster = [img.location.get('road_name') for img in segment_images if img.location and img.location.get('road_name')]
            cities_in_cluster = [img.location.get('city') for img in segment_images if img.location and img.location.get('city') and img.location.get('city') != 'Unknown']
            states_in_cluster = [img.location.get('state') for img in segment_images if img.location and img.location.get('state') and img.location.get('state') != 'Unknown']

            if road_names_in_cluster:
                most_common_road = Counter(r for r in road_names_in_cluster if r).most_common(1)
                if most_common_road: cluster_name = most_common_road[0][0]
            elif cities_in_cluster:
                most_common_city = Counter(c for c in cities_in_cluster if c).most_common(1)
                if most_common_city:
                    cluster_name = most_common_city[0][0]
                    if states_in_cluster:
                        most_common_state = Counter(s for s in states_in_cluster if s).most_common(1)
                        if most_common_state and most_common_state[0][0] != cluster_name:
                             cluster_name += f", {most_common_state[0][0]}"
            elif states_in_cluster:
                most_common_state = Counter(s for s in states_in_cluster if s).most_common(1)
                if most_common_state: cluster_name = most_common_state[0][0]

            # Calculate Defect Metrics and Summarize
            segment_damage_counts = defaultdict(int)
            total_segment_area = 0.0
            total_segment_depth = 0.0
            valid_depth_count = 0
            image_details_for_template = []

            for img in segment_images:
                metrics = calculate_defect_metrics(img)
                dmg_type = metrics.get('damage_type', 'Unknown')
                segment_damage_counts[dmg_type] += 1
                total_segment_area += metrics.get('area_m2', 0.0)
                depth = metrics.get('depth_cm', 0.0)
                if depth > 0:
                    total_segment_depth += depth
                    valid_depth_count += 1

                image_details_for_template.append({
                    'id': str(img.id),
                    'filename': img.original_filename or img.filename,
                    'type': dmg_type,
                    'area': metrics.get('area_m2', 0.0),
                    'depth': depth,
                    'severity': metrics.get('severity', 'Low'),
                    'upload_time': img.upload_time
                })

            avg_segment_depth = (total_segment_depth / valid_depth_count) if valid_depth_count > 0 else 0.0

            segment_data.append({
                'id': i + 1,
                'name': cluster_name,
                'center': segment_center,
                'image_count': len(segment_images),
                'severity': segment_severity,
                'damage_summary': dict(segment_damage_counts),
                'total_area_m2': round(total_segment_area, 2),
                'avg_depth_cm': round(avg_segment_depth, 1),
                'images': sorted(image_details_for_template, key=lambda x: x['upload_time'], reverse=True)
            })

        segment_data.sort(key=lambda x: x['name'])

        app.logger.info(f"Road Segments: Prepared data for {len(segment_data)} segments.")

        return render_template('road_segments.html',
                              has_data=True,
                              segments=segment_data,
                              cluster_distance=cluster_distance_km)

    except Exception as e:
        # Ensure traceback is imported
        import traceback
        app.logger.error(f"Error in road_segments route: {str(e)}")
        app.logger.error(traceback.format_exc())
        cluster_distance_km = float(request.args.get('distance', 10))
        return render_template('road_segments.html',
                             has_data=False,
                             error_message="An error occurred while generating road segments.",
                             show_error=True,
                             cluster_distance=cluster_distance_km)
# --- End: Road Segments Route ---
# Helper function (example - if needed elsewhere)# def some_helper():
#    pass
# Make sure this is the last function before the end of the file or other non-route code
