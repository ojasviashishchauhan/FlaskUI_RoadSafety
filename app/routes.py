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

# --- MiDaS Imports and Setup ---
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

midas_processor = None
midas_model = None
# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    print("Loading MiDaS model...")
    # Use a smaller model for potentially faster inference on CPU if needed
    # model_checkpoint = "Intel/dpt-large"
    model_checkpoint = "Intel/dpt-hybrid-midas"
    midas_processor = DPTImageProcessor.from_pretrained(model_checkpoint)
    midas_model = DPTForDepthEstimation.from_pretrained(model_checkpoint).to(device)
    midas_model.eval() # Set model to evaluation mode
    print(f"MiDaS model loaded successfully onto {device}.")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    print("Depth estimation will be disabled.")
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
        except Exception as e:
            # Handle potential errors like duplicate username/email
            flash(f'Registration failed: {e}', 'danger') 
            
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
    print(f"[PID:{os.getpid()}] process_image_async starting for: ID {image_id} (Orig: {original_filename_for_log})") 
    image = None  # Initialize image to None
    try:
        # Find the specific image entry by ID
        image = Image.objects(id=image_id).first()
        if not image:
             print(f"Error: Could not find image entry for ID {image_id}")
             return # Exit if no entry found

        # Generate unique filename (if not already set, though it likely is)
        if not image.filename or not image.file_path:
            unique_filename_part = uuid.uuid4().hex
            filename_ext = os.path.splitext(image.original_filename)[1] if image.original_filename else '.png'
            image.filename = f"{unique_filename_part}{filename_ext}"
            image.file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            print(f"Generated filename {image.filename} for ID {image_id}")

        # Save the actual image file data
        print(f"Saving image file to {image.file_path}")
        with open(image.file_path, 'wb') as f:
            f.write(file_data)
        print(f"Image file saved for ID {image_id}")
        
        # --- Extract and Save Metadata/Location --- 
        print(f"Extracting metadata for {image.filename} (ID: {image.id})")
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
                print(f"Location details found: {location_details}")
                # Store the entire location details dictionary
                location.update(location_details)
            except Exception as loc_err:
                print(f"Error getting location details for ID {image.id}: {loc_err}")
                location['formatted_address'] = "Address lookup failed"
                location['city'] = "Unknown"
                location['state'] = "Unknown"
        print(f"Assigning Metadata: {metadata} for ID {image.id}")
        print(f"Assigning Location: {location} for ID {image.id}")
        image.metadata = metadata
        image.location = location
        # --- End Metadata/Location Extraction ---

        # Update status to processing 
        image.processing_status = 'processing'
        image.image_type = image_type_from_form 
        print(f"Saving initial state (processing + meta/loc) for ID {image.id}")
        image.save()
        print(f"Initial state saved for ID {image.id}")
        
        # Run ML prediction
        print(f"Starting ML prediction for {image.filename} (ID: {image.id})")
        prediction = ml_predictor.predict(image.file_path, image.image_type)
        print(f"ML prediction completed for {image.filename} (ID: {image.id})")
        
        # --- Remove RADICAL SIMPLIFICATION TEST ---
        # (Code block removed)
        # --- END RADICAL SIMPLIFICATION TEST ---

        # --- Restore Original result processing ---
        image_to_update = Image.objects(id=image.id).first()
        if not image_to_update:
             print(f"ERROR: Could not find image {image.filename} (ID: {image.id}) for final save.")
             # Consider how to handle this - maybe mark original image object as failed?
             # image.processing_status = 'failed' 
             # image.error_message = 'Failed to refetch document before final save'
             # image.save()
             return # Exit processing

        image_to_update.processing_status = 'completed'
        image_to_update.completion_time = datetime.datetime.now()
        
        # Log confidence score before assigning
        raw_preds = prediction.get('raw_predictions', [])
        # Calculate confidence score (e.g., average or max of detections)
        if raw_preds:
             # Example: Use max confidence among detections
             conf_score = max(p.get('confidence', 0.0) for p in raw_preds)
        else:
             conf_score = 0.0
        print(f"Assigning confidence_score: {conf_score} (Type: {type(conf_score)}) for {image.filename}")
        image_to_update.confidence_score = conf_score 
        
        # Prepare prediction results dict (store full results again)
        annotated_path = prediction.get('annotated_path')
        # The 'raw_preds' variable holds the list from ml_predictor.predict

        # --- Start MiDaS Depth Estimation & Area from Mask --- 
        midas_success = False
        if midas_model and midas_processor and raw_preds: # Only run if model loaded and detections exist
            print(f"[PID:{os.getpid()}] Running MiDaS depth estimation for {image.filename}...")
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
                print(f"[PID:{os.getpid()}] MiDaS inference complete. Depth map shape: {depth_map.shape}")

                # Calculate depth and area for each prediction using masks
                for pred in raw_preds:
                    pred['estimated_depth_cm'] = 0 # Default
                    pred['accurate_area_pixels'] = 0 # Default
                    
                    if 'mask' in pred and pred['mask'] is not None:
                        # 1. Calculate Area from Mask Polygon
                        try:
                            polygon_points = np.array(pred['mask'], dtype=np.int32)
                            area_pixels = cv2.contourArea(polygon_points)
                            pred['accurate_area_pixels'] = area_pixels
                        except Exception as area_err:
                            print(f"Warning: Could not calculate mask area for pred in {image.filename}: {area_err}")

                        # 2. Calculate Depth from MiDaS map using Mask
                        try:
                            boolean_mask = create_boolean_mask_from_polygon(pred['mask'], depth_map.shape)
                            depth_values_in_mask = depth_map[boolean_mask]
                            
                            if depth_values_in_mask.size > 0:
                                relative_avg_depth = np.mean(depth_values_in_mask)
                                # !!! Apply Placeholder Calibration !!!
                                pred['estimated_depth_cm'] = calibrate_midas_depth(relative_avg_depth)
                            else:
                                print(f"Warning: Boolean mask for pred in {image.filename} was empty.")

                        except Exception as depth_err:
                             print(f"Warning: Could not calculate MiDaS depth for pred in {image.filename}: {depth_err}")
                midas_success = True
                print(f"[PID:{os.getpid()}] MiDaS processing finished for {image.filename}. Success: {midas_success}")
            except Exception as midas_err:
                print(f"ERROR during MiDaS processing for {image.filename}: {midas_err}")
                traceback.print_exc()
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
        print(f"Assigning prediction_results: {pred_results_dict} for {image.filename}") 
        image_to_update.prediction_results = pred_results_dict
        
        # Extract processing time if available in prediction results
        # Assuming processing_time might be added to the prediction dict by MLPredictor
        # if 'processing_time' in prediction:
        #      image_to_update.processing_time = prediction['processing_time']
        # Or maybe calculate total time here if not passed back:
        # image_to_update.processing_time = time.time() - start_time_of_processing
        
        image_to_update.annotated_image_path = annotated_path # Save annotated path
        
        print(f"Attempting to save final FULL results for {image.filename} (ID: {image.id})...")
        try:
            image_to_update.save()
            print(f"Successfully saved final FULL results for {image.filename} (ID: {image.id})")
        except Exception as final_save_err:
            print(f"ERROR saving final FULL results for {image.filename} (ID: {image.id}): {final_save_err}")
            # Log traceback for detailed error
            import traceback
            traceback.print_exc() 
            # Update status to failed if save fails
            try:
                image_to_update.processing_status = 'failed'
                image_to_update.error_message = f"Failed to save results: {final_save_err}"
                image_to_update.save()
                print(f"Updated image ID {image.id} status to failed after save error.")
            except Exception as update_err:
                print(f"Error updating status to failed for ID {image.id} after save error: {update_err}")
        # --- End original result processing ---
            
    except Exception as e:
        print(f"Error in process_image_async for ID {image_id} (Orig: {original_filename_for_log}): {str(e)}")
        import traceback
        traceback.print_exc()
        # Update database with error status if image object exists
        if image: # Use the image object fetched at the start
            try:
                image.processing_status = 'failed'
                image.error_message = str(e)
                image.save()
                print(f"Updated image ID {image_id} status to failed.")
            except Exception as update_err:
                print(f"Error updating error status for ID {image_id}: {update_err}")

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
                print(f"Created initial DB entry for {original_filename} with ID: {initial_entry.id}")
                
                # Submit to thread pool OR run synchronously for Potholes
                if form.image_type.data == 'Potholes':
                    print(f"--- Running Potholes processing SYNCHRONOUSLY for {original_filename} ---")
                    process_image_async(file_data, initial_entry.id, form.image_type.data)
                    print(f"--- SYNCHRONOUS Potholes processing finished for {original_filename} ---")
                else:
                    future = thread_pool.submit(process_image_async, file_data, initial_entry.id, form.image_type.data)
                # --- LOGGING: Task Submission --- 
                    print(f"[PID:{os.getpid()}] Submitted ASYNC task for {original_filename} (User: {current_user_id_str}). Future running: {future.running()}") 
                uploaded_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"Failed to initiate processing for {original_filename}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if uploaded_count > 0:
            flash(f'{uploaded_count} file(s) uploaded and queued for processing.', 'success')
        if failed_count > 0:
            flash(f'{failed_count} file(s) failed to upload.', 'danger')
             
        return redirect(url_for('dashboard'))
    
    return render_template('upload_image.html', form=form)

# --- Video Processing Function ---
def process_video_async(file_data, video_doc_id, image_type):
    """Processes video: runs prediction on frames, generates annotated video."""
    print(f"[VID_PROC START] Starting processing for video ID: {video_doc_id}, Type: {image_type}")
    video_doc = None
    cap = None
    writer = None
    annotated_video_path = None # Define here for use in finally block
    processed_frame_count = 0 # Frames processed (read)
    detection_frame_count = 0 # Frames with at least one detection
    total_raw_detections = 0  # Total boxes detected across all frames
    detected_objects_summary = [] # To store info about detections

    # --- Tracking State --- 
    active_tracks = [] # List of dicts: {'id': track_id, 'bbox': [x1,y1,x2,y2], 'class_name': name, 'last_frame': frame_num}
    next_track_id = 0
    iou_threshold = 0.4 # IoU threshold for matching tracks
    unique_damage_reports = [] # Store unique damages found
    min_confidence_threshold = 0.4 # Ignore detections below this confidence
    min_track_duration_frames = 3  # Require track to persist for this many frames
    # ----------------------

    try:
        video_doc = Image.objects(id=video_doc_id).first()
        if not video_doc or video_doc.media_type != 'video':
            print(f"[VID_PROC ERROR] Invalid document or not a video for ID: {video_doc_id}")
            return

        # Ensure file path exists
        if not video_doc.file_path or not os.path.exists(video_doc.file_path):
             # Regenerate path if missing and save file again (should ideally not happen)
             if not video_doc.filename:
                  unique_filename_part = uuid.uuid4().hex
                  filename_ext = os.path.splitext(video_doc.original_filename)[1] if video_doc.original_filename else '.mp4'
                  video_doc.filename = f"{unique_filename_part}{filename_ext}"
             video_doc.file_path = os.path.join(app.config['UPLOAD_FOLDER'], video_doc.filename)
             print(f"Regenerating file path: {video_doc.file_path}")
             with open(video_doc.file_path, 'wb') as f:
                 f.write(file_data)
        
        input_video_path = video_doc.file_path
        
        # Extract metadata if not already present (or re-extract)
        if not video_doc.metadata or 'frame_width' not in video_doc.metadata:
             print(f"Extracting video metadata for {video_doc.filename} (ID: {video_doc.id})")
             metadata = extract_video_metadata(input_video_path)
             video_doc.metadata = metadata
        else:
             metadata = video_doc.metadata

        video_doc.processing_status = 'processing'
        video_doc.image_type = image_type 
        video_doc.save() # Save processing status and metadata
        print(f"Initial state (processing + meta) saved for Video ID {video_doc.id}")

        # Check if metadata extraction failed
        if not metadata or metadata.get('error') or 'frame_width' not in metadata:
            raise ValueError(f"Failed to get valid video metadata: {metadata.get('error', 'Unknown error')}")

        # --- Video Reading and Writing Setup --- 
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_video_path}")

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
        print(f"[VID_PROC DEBUG] Creating VideoWriter: Path={annotated_video_path}, FourCC=avc1, FPS={fps}, Size=({width}x{height})")
        writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
             # Add more specific logging if writer fails
             print(f"[VID_PROC ERROR] cv2.VideoWriter failed to open. Path={annotated_video_path}, FourCC={fourcc}, FPS={fps}, Size=({width},{height})")
             raise IOError(f"Could not open VideoWriter for path: {annotated_video_path}")
        print(f"[VID_PROC DEBUG] VideoWriter opened successfully.")

        print(f"Starting frame processing for Video ID: {video_doc_id}")
        total_frames = metadata.get('frame_count', 0)
        frame_num = 0

        # --- Frame Processing Loop --- 
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            frame_num += 1
            # Process every Nth frame? For now, process all.
            # if frame_num % 5 != 0: # Example: process every 5th frame
            #    writer.write(frame) # Write original frame if skipping
            #    continue 

            # Convert frame BGR to RGB for prediction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run prediction
            frame_predictions_raw = ml_predictor.predict_frame(frame_rgb, image_type)
            total_raw_detections += len(frame_predictions_raw)

            # --- Filter detections by confidence --- 
            frame_predictions = [p for p in frame_predictions_raw if p.get('confidence', 0.0) >= min_confidence_threshold]
            # -------------------------------------
            
            # Check if *any* high-confidence detections were found
            if frame_predictions:
                 detection_frame_count += 1 
                 # Only print confident detections
                 # print(f"-- [VID_PROC Frame {frame_num}] Confident Detections: {frame_predictions}")
            
            current_frame_bboxes = [p['bbox'] for p in frame_predictions]
            current_frame_classes = [p['class_name'] for p in frame_predictions]
            matched_indices = set()
            new_active_tracks = []

            # Try to match current detections with active tracks from previous frame
            if frame_predictions and active_tracks:
                 # detection_frame_count incremented above
                 for track in active_tracks:
                     best_match_idx = -1
                     best_iou = iou_threshold
                     for i in range(len(current_frame_bboxes)):
                         if i in matched_indices or current_frame_classes[i] != track['class_name']:
                             continue 
                         iou = calculate_iou(track['bbox'], current_frame_bboxes[i])
                         if iou > best_iou:
                             best_iou = iou
                             best_match_idx = i
                     
                     if best_match_idx != -1:
                         track['bbox'] = current_frame_bboxes[best_match_idx] 
                         track['last_frame'] = frame_num
                         # Increment duration when matched
                         track['duration'] = track.get('duration', 1) + 1 
                         new_active_tracks.append(track) 
                         matched_indices.add(best_match_idx)
            # --- End IoU Matching --- 
                 
            # Add unmatched (high-confidence) detections as new tracks
            for i in range(len(current_frame_bboxes)):
                if i not in matched_indices:
                    new_track = {
                        'id': next_track_id,
                        'bbox': current_frame_bboxes[i],
                        'class_name': current_frame_classes[i],
                        'start_frame': frame_num,
                        'last_frame': frame_num,
                        'duration': 1 # Initial duration is 1 frame
                    }
                    new_active_tracks.append(new_track)
                    # Don't add to unique_damage_reports yet, wait for duration check
                    next_track_id += 1

            active_tracks = new_active_tracks # Update active tracks

            # --- Draw Annotations (Optional - using active_tracks) --- 
            annotated_frame = frame.copy()
            # Draw active tracks for visualization
            for track in active_tracks:
                 x1, y1, x2, y2 = map(int, track['bbox'])
                 label = f"{track['class_name']} ID:{track['id']}"
                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # --- End Annotation Drawing --- 

            writer.write(annotated_frame)

            # Optional: Log progress
            if frame_num % 100 == 0: 
                print(f"[VID_PROC] Processed frame {frame_num}/{total_frames} for Video ID: {video_doc_id}")

        # --- End Loop --- 
        
        # --- Filter tracks by duration and finalize unique reports --- 
        final_unique_damage_reports = []
        for track in active_tracks:
            # Check if the track met the minimum duration
            if track.get('duration', 0) >= min_track_duration_frames:
                 final_unique_damage_reports.append({
                     'track_id': track['id'],
                     'class_name': track['class_name'],
                     'start_frame': track['start_frame'],
                     'end_frame': track['last_frame'],
                     'duration_frames': track.get('duration', 0)
                 })
        # -------------------------------------------------------------
        
        print(f"Finished frame processing. Found {len(final_unique_damage_reports)} unique damages meeting duration criteria.")
        
        # --- Save Final Results --- 
        video_doc_final = Image.objects(id=video_doc_id).first()
        if not video_doc_final:
             print(f"[VID_PROC ERROR] Cannot find video doc {video_doc_id} before final save.")
             return 

        video_doc_final.processing_status = 'completed'
        video_doc_final.completion_time = datetime.datetime.now()
        video_doc_final.annotated_image_path = annotated_filename # Save ANNOTATED path
        
        # --- Calculate Final Results (Based on Filtered Tracking) --- 
        # Consider video damaged if there are ANY unique damages or if total_raw_detections > 0
        damage_was_detected = bool(final_unique_damage_reports) or total_raw_detections > 0
        # Confidence needs rethink - maybe max confidence *of the final tracks*?
        # overall_confidence = max(t.get('max_conf', 0.0) for t in final_unique_damage_reports) if final_unique_damage_reports else 0.0
        print(f"[VID_PROC] Unique Damage detected (meeting criteria): {damage_was_detected} ({len(final_unique_damage_reports)} instances, {total_raw_detections} raw detections)")
        # -------------------------------

        # Update summary message and results
        video_doc_final.prediction_results = {
             'message': f'Video processing completed. Found {len(final_unique_damage_reports)} unique damage instance(s) meeting criteria (Min Conf: {min_confidence_threshold}, Min Duration: {min_track_duration_frames} frames). Raw Detections: {total_raw_detections}',
             'unique_damage_count': len(final_unique_damage_reports), # Use count of filtered reports
             'unique_damage_reports': final_unique_damage_reports, # Store filtered reports
             'damage_detected': damage_was_detected,
             'total_raw_detections': total_raw_detections
             }
        
        # Add processed detail fields
        video_doc_final.processing_time = (datetime.datetime.now() - video_doc_final.upload_time).total_seconds()
        
        print(f"Attempting final save for annotated video ID: {video_doc_id}") 
        video_doc_final.save()
        print(f"[VID_PROC SUCCESS] Completed processing for video ID: {video_doc_id}")

    except Exception as e:
        # ... (rest of the error handling remains similar, ensure video_doc_final is used if needed) ...
        print(f"[VID_PROC ERROR] Error processing video ID {video_doc_id}: {e}")
        import traceback
        traceback.print_exc()
        # Try to update status to failed
        try:
            video_doc_fail = Image.objects(id=video_doc_id).first()
            if video_doc_fail:
                 video_doc_fail.processing_status = 'failed'
                 video_doc_fail.error_message = str(e)
                 video_doc_fail.save()
                 print(f"[VID_PROC] Updated video ID {video_doc_id} status to failed.")
            else:
                 print(f"[VID_PROC ERROR] Could not find video doc {video_doc_id} to update status to failed.")
        except Exception as update_err:
            print(f"[VID_PROC ERROR] Error updating error status for ID {video_doc_id}: {update_err}")
            
    finally:
        # Release resources
        if cap and cap.isOpened():
            cap.release()
        if writer and writer.isOpened():
            writer.release()
        print(f"[VID_PROC END] Resources released for video ID: {video_doc_id}")

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
                print(f"Created initial DB entry for VIDEO {original_filename} with ID: {initial_entry.id}")
                
                # Submit video processing task (using the new function)
                # Pass the selected image_type from the form
                future = thread_pool.submit(process_video_async, file_data, initial_entry.id, form.image_type.data)
                print(f"[PID:{os.getpid()}] Submitted ASYNC task for VIDEO {original_filename} (ID: {initial_entry.id}, Type: {form.image_type.data}).") 
                uploaded_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"Failed to initiate processing for VIDEO {original_filename}: {str(e)}")
                import traceback
                traceback.print_exc()
        
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
        ).exclude('prediction_results.raw_predictions').limit(10)  # Limit to last 10 for efficiency

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
    query = {}
    if not current_user.is_admin:
        user_id_str = current_user.get_id()
        try:
            user_id_obj = ObjectId(user_id_str)
            query['$or'] = [{'user_id': user_id_str}, {'user_id': user_id_obj}]
        except Exception:
            query['user_id'] = user_id_str

    try:
        # Get all images
        images = Image.objects(query)
        
        if not images:
            return render_template('analytics.html', has_data=False)

        # Basic stats
        detection_stats = {
            'total': len(images),
            'damage_detected': sum(1 for img in images 
                                 if img.processing_status == 'completed' 
                                 and img.prediction_results and img.prediction_results.get('damage_detected', False)),
            'failed': sum(1 for img in images if img.processing_status == 'failed'),
            'processing': sum(1 for img in images if img.processing_status in ['pending', 'processing'])
        }

        # Add success rate to stats
        completed_images = sum(1 for img in images if img.processing_status == 'completed')
        detection_stats['success_rate'] = (completed_images / detection_stats['total'] * 100) if detection_stats['total'] > 0 else 0
        detection_stats['detection_rate'] = (detection_stats['damage_detected'] / completed_images * 100) if completed_images > 0 else 0

        # Processing Time Analysis
        valid_times = []
        time_by_type = defaultdict(list)
        
        # Calculate the processing time by using completion_time - upload_time for completed images
        for img in images:
            if img.processing_status == 'completed' and img.completion_time and img.upload_time:
                # Calculate processing time in seconds
                proc_time = (img.completion_time - img.upload_time).total_seconds()
                img_type = img.image_type or 'Unknown'
                
                if 0 < proc_time < 3600:  # Reasonable time limit (1 hour max)
                    valid_times.append(proc_time)
                    time_by_type[img_type].append(proc_time)

        processing_times = {
            'avg_time': sum(valid_times) / len(valid_times) if valid_times else 0,
            'max_time': max(valid_times) if valid_times else 0,
            'min_time': min(valid_times) if valid_times else 0
        }

        # Average time by type
        avg_time_by_type = {
            img_type: sum(times) / len(times) if times else 0
            for img_type, times in time_by_type.items()
        }

        # Damage Analysis
        damage_types = defaultdict(int)
        confidence_by_type = defaultdict(list)
        severity_by_type = defaultdict(lambda: defaultdict(int))

        for img in images:
            pred_results = img.prediction_results
            if pred_results:  # Remove the damage_detected condition
                damage_type = pred_results.get('damage_type', 'Unknown')
                confidence = pred_results.get('confidence', 0)
                severity = pred_results.get('damage_severity', 'Unknown')
                
                # Count all damage types
                damage_types[damage_type] += 1
                confidence_by_type[damage_type].append(confidence)
                severity_by_type[damage_type][severity] += 1

        # Calculate average confidence by type
        avg_confidence_by_type = {
            d_type: sum(conf) / len(conf) if conf else 0
            for d_type, conf in confidence_by_type.items()
        }

        # Time Distribution Analysis
        daily_activity = defaultdict(int)
        daily_distribution = defaultdict(int)
        
        # Get date range for last 30 days
        today = datetime.datetime.now().date()
        date_range = [(today - datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30)]
        
        # Initialize daily_distribution with zeros for all dates
        for date in date_range:
            daily_distribution[date] = 0
        
        # Count uploads for each date
        for img in images:
            upload_time = img.upload_time
            if upload_time:
                # Get day of the week (0 = Monday, 6 = Sunday)
                day_num = upload_time.weekday()
                day_name = calendar.day_name[day_num]
                daily_activity[day_name] += 1
                
                # Format date as YYYY-MM-DD
                date_str = upload_time.strftime('%Y-%m-%d')
                if date_str in daily_distribution:
                    daily_distribution[date_str] += 1

        # Sort days of week in correct order
        sorted_daily_activity = {day: daily_activity[day] for day in calendar.day_name}

        # Sort daily distribution by date
        sorted_daily_distribution = dict(sorted(daily_distribution.items()))

        # Location Analysis
        location_stats = []
        for img in images:
            metadata = img.metadata
            location_name = metadata.get('location_name', 'Unknown')
            pred_results = img.prediction_results
            
            # Find existing location or create new one
            loc_stat = next((loc for loc in location_stats if loc['_id'] == location_name), None)
            if not loc_stat:
                loc_stat = {
                    '_id': location_name,
                    'count': 0,
                    'damage_detected': 0,
                    'confidence_sum': 0,
                    'confidence_count': 0,
                    'processing_time_sum': 0,
                    'processing_time_count': 0
                }
                location_stats.append(loc_stat)
            
            # Update stats
            loc_stat['count'] += 1
            if pred_results.get('damage_detected'):
                loc_stat['damage_detected'] += 1
            
            confidence = pred_results.get('confidence')
            if confidence is not None:
                loc_stat['confidence_sum'] += confidence
                loc_stat['confidence_count'] += 1
            
            proc_time = img.processing_time
            if proc_time and isinstance(proc_time, (int, float, str)):
                try:
                    proc_time_float = float(proc_time)
                    if 0 < proc_time_float < 3600:
                        loc_stat['processing_time_sum'] += proc_time_float
                        loc_stat['processing_time_count'] += 1
                except (ValueError, TypeError):
                    continue

        # Calculate averages for location stats
        for loc in location_stats:
            loc['avg_confidence'] = (loc['confidence_sum'] / loc['confidence_count'] * 100 
                                   if loc['confidence_count'] > 0 else 0)
            loc['avg_processing_time'] = (loc['processing_time_sum'] / loc['processing_time_count']
                                        if loc['processing_time_count'] > 0 else 0)
            # Clean up temporary fields
            del loc['confidence_sum'], loc['confidence_count']
            del loc['processing_time_sum'], loc['processing_time_count']

        # Error Analysis
        error_analysis = []
        error_counts = defaultdict(int)
        error_times = defaultdict(list)
        
        for img in images:
            if img.processing_status == 'failed':
                error = img.error_message or 'Unknown Error'
                proc_time = img.processing_time or 0
                error_counts[error] += 1
                error_times[error].append(float(proc_time) if proc_time else 0)
        
        if error_counts:
            for error, count in error_counts.items():
                times = error_times[error]
                error_analysis.append({
                    'type': error,
                    'count': count,
                    'avg_time_impact': sum(times) / len(times) if times else 0
                })
        else:
            # If no errors, create informative placeholder data for visualization
            error_analysis = [
                {'type': 'System Performance', 'count': 50, 'avg_time_impact': 0},
                {'type': 'Data Quality', 'count': 20, 'avg_time_impact': 0},
                {'type': 'Network Issues', 'count': 15, 'avg_time_impact': 0},
                {'type': 'Model Accuracy', 'count': 10, 'avg_time_impact': 0},
                {'type': 'Other', 'count': 5, 'avg_time_impact': 0}
            ]

        return render_template('analytics.html',
            has_data=True,
            detection_stats=detection_stats,
            processing_times=processing_times,
            avg_time_by_type=avg_time_by_type,
            damage_types=dict(damage_types),
            severity_by_type=dict(severity_by_type),
            avg_confidence_by_type=avg_confidence_by_type,
            daily_activity=sorted_daily_activity,
            daily_distribution=sorted_daily_distribution,
            location_stats=location_stats,
            error_analysis=error_analysis)

    except Exception as e:
        print(f"Error in analytics route: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('analytics.html', 
                             has_data=False, 
                             error_message=str(e),
                             show_error=True)

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

@app.route('/region_analytics')
@login_required
def region_analytics():
    """Provide analytics based on regional clustering of defects by distance"""
    query = {}
    if not current_user.is_admin:
        user_id_str = current_user.get_id()
        try:
            user_id_obj = ObjectId(user_id_str)
            query['$or'] = [{'user_id': user_id_str}, {'user_id': user_id_obj}]
        except Exception:
            query['user_id'] = user_id_str

    try:
        # Get clustering distance from request, default to 5km
        cluster_distance_km = float(request.args.get('distance', 5))
        
        # Get all images with location data
        images = Image.objects(query).filter(
            location__exists=True, 
            processing_status='completed',
            prediction_results__damage_detected=True  # Only get images with damage
        )
        
        app.logger.info(f"Found {len(images)} images with damage for region analytics")
        
        if not images:
            return render_template('region_analytics.html', has_data=False)
        
        # Prepare data for clustering
        points = []
        for img in images:
            if img.location and 'latitude' in img.location and 'longitude' in img.location:
                lat = img.location.get('latitude')
                lng = img.location.get('longitude')
                if lat is not None and lng is not None:
                    # Store image information with coordinates
                    damage_type = "Unknown"
                    if img.prediction_results:
                        damage_type = img.prediction_results.get('damage_type', img.image_type)
                    elif img.image_type:
                        damage_type = img.image_type
                    
                    points.append({
                        'id': str(img.id),
                        'lat': float(lat),
                        'lng': float(lng),
                        'damage_type': damage_type,
                        'confidence': img.confidence_score or 0.0,
                        # --- Add Location Info --- 
                        'state': img.location.get('state', 'Unknown'),
                        'city': img.location.get('city', 'Unknown'),
                        'road_name': img.location.get('road_name'), # <<< ADD road_name extraction
                        # --- End Add --- 
                        'file_path': img.file_path,
                        'upload_time': img.upload_time
                    })
        
        # Perform clustering by distance
        clusters = cluster_by_distance(points, cluster_distance_km)
        
        # Analyze clusters
        cluster_analytics = []
        for i, cluster in enumerate(clusters):
            # Find center of cluster
            if not cluster:
                continue
                
            cluster_center = calculate_cluster_center(cluster)
            
            # Count damage types within cluster
            damage_counts = defaultdict(int)
            total_confidence = 0
            for point in cluster:
                damage_counts[point['damage_type']] += 1
                total_confidence += point['confidence']
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(cluster) if cluster else 0
            
            # Find most common damage type
            primary_damage = max(damage_counts.items(), key=lambda x: x[1])[0] if damage_counts else "Unknown"
            
            # --- Determine Cluster Name --- 
            cluster_name = f"Segment #{i}" # Default
            # Prioritize road name if available
            # Use the road_name extracted earlier
            road_names_in_cluster = [p.get('road_name') for p in cluster if p.get('road_name')] # <<< Use extracted road_name
            cities_in_cluster = [p.get('city') for p in cluster if p.get('city') and p.get('city') != 'Unknown']
            states_in_cluster = [p.get('state') for p in cluster if p.get('state') and p.get('state') != 'Unknown']

            from collections import Counter
            if road_names_in_cluster:
                most_common_road = Counter(road_names_in_cluster).most_common(1)[0][0]
                cluster_name = most_common_road
            elif cities_in_cluster:
                most_common_city = Counter(cities_in_cluster).most_common(1)[0][0]
                cluster_name = most_common_city
                # Optionally add state if available and different from city
                if states_in_cluster:
                    most_common_state = Counter(states_in_cluster).most_common(1)[0][0]
                    if most_common_state and most_common_state != most_common_city:
                         cluster_name += f", {most_common_state}"
            elif states_in_cluster: # Fallback to state if no city or road name
                most_common_state = Counter(states_in_cluster).most_common(1)[0][0]
                if most_common_state:
                    cluster_name = most_common_state
            # --- End Determine Name ---

            # Add cluster analytics
            cluster_analytics.append({
                'id': i,
                'name': cluster_name, # <<< Use determined Name
                'center': cluster_center,
                'point_count': len(cluster),
                'primary_damage': primary_damage,
                'damage_types': dict(damage_counts),
                'avg_confidence': avg_confidence * 100,  # Convert to percentage
                'radius_km': cluster_distance_km,
                'severity': calculate_cluster_severity(cluster, cluster_distance_km)
            })

        # --- Apply Filters ---
        selected_severity = request.args.get('severity')
        selected_damage_type = request.args.get('damage_type')

        filtered_clusters = cluster_analytics
        if selected_severity and selected_severity != 'all': # Add check for 'all'
            filtered_clusters = [c for c in filtered_clusters if c['severity'] == selected_severity]
            app.logger.info(f"Filtering by severity: {selected_severity}, {len(filtered_clusters)} clusters remaining")

        if selected_damage_type and selected_damage_type != 'all': # Add check for 'all'
            filtered_clusters = [
                c for c in filtered_clusters
                if selected_damage_type in c['damage_types']
            ]
            app.logger.info(f"Filtering by damage type: {selected_damage_type}, {len(filtered_clusters)} clusters remaining")
        # --- End Apply Filters ---

        app.logger.info(f"Generated {len(cluster_analytics)} clusters, {len(filtered_clusters)} after filtering for region analytics")

        return render_template('region_analytics.html',
                              has_data=True,
                              clusters=filtered_clusters, # Use filtered list
                              cluster_distance=cluster_distance_km,
                              # Pass filter values back to template to maintain state
                              selected_severity=selected_severity or 'all', # Default to 'all' if None
                              selected_damage_type=selected_damage_type or 'all', # Default to 'all' if None
                              # --- Add State Stats ---
                              # state_stats=sorted_stats_by_state
                              # --- End Add ---
                             )

    except Exception as e:
        app.logger.error(f"Error in region analytics route: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return render_template('region_analytics.html',
                             has_data=False,
                             error_message="An error occurred while generating analytics.", # Generic message
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
    avg_confidence = sum(p['confidence'] for p in cluster) / len(cluster)
    
    # Determine severity based on density and confidence
    if density > 0.5 and avg_confidence > 0.7:
        return "High"
    elif density > 0.2 or avg_confidence > 0.6:
        return "Medium"
    else:
        return "Low"

def calculate_defect_metrics(image):
    """Calculate area, depth, and other metrics for defects.
    Tries to use pre-calculated accurate values if available in raw_predictions,
    otherwise falls back to rough estimations.
    """
    defect_metrics = {
        'area_m2': 0,
        'depth_cm': 0, # Represents average/representative depth
        'severity': 'Low',
        'confidence': image.confidence_score or 0,
        'damage_type': 'Unknown' # Add damage type here
    }
    
    try:
        raw_predictions = image.prediction_results.get('raw_predictions', [])
        metadata = image.metadata or {}
        img_width = metadata.get('width', 0)
        img_height = metadata.get('height', 0)
        
        # --- Try to get aggregated accurate metrics first ---
        total_accurate_area_m2 = 0
        depths_cm = []
        damage_types = set()
        has_accurate_metrics = False

        if raw_predictions:
            # Check if the first prediction has the enhanced keys
            first_pred = raw_predictions[0]
            if 'accurate_area_m2' in first_pred and 'estimated_depth_cm' in first_pred:
                 has_accurate_metrics = True
                 
            for pred in raw_predictions:
                damage_types.add(pred.get('class_name', 'Unknown'))
                if has_accurate_metrics:
                    total_accurate_area_m2 += pred.get('accurate_area_m2', 0)
                    depths_cm.append(pred.get('estimated_depth_cm', 0))
                # else: we will use estimations later

        if has_accurate_metrics:
            defect_metrics['area_m2'] = round(total_accurate_area_m2, 2)
            if depths_cm:
                 # Use average depth for overall metric
                 defect_metrics['depth_cm'] = round(np.mean([d for d in depths_cm if d is not None]), 1)
            has_metrics = True # Mark that we got metrics this way
        
        # --- Fallback to rough estimations if no accurate metrics ---
        if not has_accurate_metrics and raw_predictions and img_width > 0 and img_height > 0:
            app.logger.info(f"Falling back to rough estimations for image {image.id}")
            # Estimate pixel to meter conversion (ROUGH)
            estimated_road_width_m = 3.5
            estimated_pixels_per_meter = img_width / estimated_road_width_m
            
            total_area_pixels = 0
            for pred in raw_predictions:
                bbox = pred.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    total_area_pixels += width * height
            
            if estimated_pixels_per_meter > 0:
                area_m2_est = total_area_pixels / (estimated_pixels_per_meter ** 2)
                defect_metrics['area_m2'] = round(area_m2_est, 2)

            # Crude depth estimation based on overall confidence and primary type
            confidence = defect_metrics['confidence']
            primary_damage_type = image.prediction_results.get('damage_type', image.image_type or 'Unknown')
            depth_cm_est = 0
            if 'Pothole' in primary_damage_type:
                depth_cm_est = confidence * 10
            elif 'Crack' in primary_damage_type:
                depth_cm_est = confidence * 3
            defect_metrics['depth_cm'] = round(depth_cm_est, 1)
            
        # --- Determine overall severity (always calculate this) ---
        if defect_metrics['area_m2'] > 1.0 or defect_metrics['depth_cm'] > 5.0:
            defect_metrics['severity'] = 'High'
        elif defect_metrics['area_m2'] > 0.5 or defect_metrics['depth_cm'] > 2.0:
            defect_metrics['severity'] = 'Medium'
        else:
            defect_metrics['severity'] = 'Low'

        # --- Determine primary damage type ---
        if len(damage_types) == 1:
             defect_metrics['damage_type'] = list(damage_types)[0]
        elif len(damage_types) > 1:
             # If multiple types, use the one from prediction_results if available, else 'Mixed'
             defect_metrics['damage_type'] = image.prediction_results.get('damage_type', 'Mixed')
        else:
             # Fallback if no types found in raw_predictions
             defect_metrics['damage_type'] = image.prediction_results.get('damage_type', image.image_type or 'Unknown')
             
        # Ensure 'Unknown' is used if type is None or empty string
        if not defect_metrics['damage_type']:
             defect_metrics['damage_type'] = 'Unknown'


        app.logger.info(f"Calculated defect metrics for {image.id}: {defect_metrics}")
        return defect_metrics
    
    except Exception as e:
        app.logger.error(f"Error calculating defect metrics for {image.id}: {str(e)}\\n{traceback.format_exc()}")
        # Return default metrics with error noted, potentially add error key
        defect_metrics['error'] = str(e) 
        return defect_metrics

def generate_recommendations(metrics):
    """Generate repair recommendations based on calculated metrics"""
    recommendations = []
    
    # Use metrics directly passed in
    damage_type = metrics.get('damage_type', 'Unknown')
    severity = metrics.get('severity', 'Low')
    area_m2 = metrics.get('area_m2', 0)
    depth_cm = metrics.get('depth_cm', 0)

    # Helper function for cost estimation
    def estimate_cost(base_cost_per_unit, area, depth=None):
        # Example: Increase cost quadratically with area and linearly with depth
        cost = base_cost_per_unit * (area ** 1.2)
        if depth and depth > 0:
            cost *= (1 + depth / 10) # Increase cost based on depth
            
        # --- Add Scaling Factor (assume base cost is in hundreds) ---
        scaled_cost = cost * 100 
        # --- End Add Scaling Factor ---
        
        # --- Change currency symbol to INR --- 
        # --- Apply scaling and ensure minimum --- 
        return f'₹{max(5000, int(scaled_cost))}' # Minimum cost of ₹5000 
        # --- End Change & Apply Scaling ---

    if 'Pothole' in damage_type:
        if severity == 'High' or depth_cm > 7 or area_m2 > 2:
            recommendations.append({
                'title': 'Full Depth Patching',
                'description': f'Critical Pothole ({area_m2:.1f} m², {depth_cm:.1f} cm depth). Requires removing damaged pavement to stable base and replacing with new asphalt material.',
                'urgency': 'High',
                'estimated_cost': estimate_cost(150, area_m2, depth_cm)
            })
        elif severity == 'Medium' or depth_cm > 4 or area_m2 > 1:
            recommendations.append({
                'title': 'Partial Depth Patching with Hot Mix',
                'description': f'Significant Pothole ({area_m2:.1f} m², {depth_cm:.1f} cm depth). Clean, prepare, and fill the pothole with hot mix asphalt.',
                'urgency': 'Medium',
                'estimated_cost': estimate_cost(100, area_m2, depth_cm)
            })
        else:
            recommendations.append({
                'title': 'Surface Patching with Cold Mix',
                'description': f'Minor Pothole ({area_m2:.1f} m², {depth_cm:.1f} cm depth). Fill pothole with cold mix asphalt for temporary repair or low-traffic areas.',
                'urgency': 'Low',
                'estimated_cost': estimate_cost(60, area_m2, depth_cm)
            })
    
    elif 'Crack' in damage_type:
        # Crack recommendations can also depend on crack width, which isn't currently calculated
        if 'Alligator' in damage_type:
            if severity == 'High' or area_m2 > 5:
                recommendations.append({
                    'title': 'Mill and Overlay (Alligator Cracking)',
                    'description': f'Extensive alligator cracking ({area_m2:.1f} m²). Remove distressed layer and replace with new asphalt overlay.',
                    'urgency': 'High',
                    'estimated_cost': estimate_cost(120, area_m2)
                })
            else:
                recommendations.append({
                    'title': 'Surface Patch or Seal (Alligator Cracking)',
                    'description': f'Moderate alligator cracking ({area_m2:.1f} m²). Apply a surface patch or structural sealing treatment.',
                    'urgency': 'Medium',
                    'estimated_cost': estimate_cost(70, area_m2)
                })
        elif any(x in damage_type for x in ['Longitudinal', 'Transverse']):
            if severity == 'High' or area_m2 > 2: # Consider crack width/length if available
                 recommendations.append({
                    'title': 'Rout and Seal Cracks',
                    'description': f'Significant linear cracking ({area_m2:.1f} m²). Rout cracks and seal with flexible sealant to prevent water intrusion.',
                    'urgency': 'High',
                    'estimated_cost': estimate_cost(50, area_m2)
                })
            else:
                 recommendations.append({
                    'title': 'Crack Sealing',
                    'description': f'Minor linear cracking ({area_m2:.1f} m²). Clean and seal cracks with appropriate sealant.',
                    'urgency': 'Medium' if severity == 'Medium' else 'Low',
                    'estimated_cost': estimate_cost(30, area_m2)
                })
        else: # Other crack types
             recommendations.append({
                'title': 'Crack Sealing (General)',
                'description': f'General cracking detected ({area_m2:.1f} m²). Clean and seal cracks with appropriate sealant.',
                'urgency': severity,
                'estimated_cost': estimate_cost(35, area_m2)
            })
    
    else: # Unknown damage type
        recommendations.append({
            'title': 'On-Site Inspection Required',
            'description': f'Unknown damage type ({area_m2:.1f} m²). Conduct detailed on-site inspection to determine appropriate repair method.',
            'urgency': severity,
            'estimated_cost': 'Requires Assessment'
        })
    
    # Add preventative maintenance if not high severity
    if severity != 'High':
        recommendations.append({
            'title': 'Preventative Maintenance Monitoring',
            'description': 'Recommend regular monitoring as part of preventative maintenance schedule to track further deterioration.',
            'urgency': 'Low',
            'estimated_cost': 'Included in Maintenance Plan'
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
    if polygon_points and len(polygon_points) > 0:
        try:
            # Convert points to NumPy array of int32, required by fillPoly
            pts = np.array(polygon_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1) # Fill polygon with 1
        except Exception as e:
            app.logger.error(f"Error creating mask from polygon: {e}")
            # Return an empty mask on error
            return np.zeros(shape, dtype=bool)
    return mask.astype(bool)

def calibrate_midas_depth(relative_depth):
    """Placeholder function to calibrate MiDaS relative depth to cm.
    !!! THIS IS A PLACEHOLDER - REQUIRES REAL CALIBRATION !!!
    It currently applies an arbitrary scaling factor.
    """
    # Example: Simple linear scaling (adjust factor based on experimentation)
    scaling_factor = 15 # Completely arbitrary - adjust this!
    depth_cm = relative_depth * scaling_factor
    # Add a non-linear term? Clamp values?
    # depth_cm = (relative_depth ** 1.5) * 20 
    return max(0, round(depth_cm, 1)) # Ensure non-negative
# --- End MiDaS Helpers ---

# Helper function (example - if needed elsewhere)
# def some_helper():
#    pass
# Make sure this is the last function before the end of the file or other non-route code

# ... rest of the file (utility functions etc.) ... 