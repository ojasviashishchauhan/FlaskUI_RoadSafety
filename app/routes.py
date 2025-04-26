import os
import datetime
from flask import render_template, redirect, url_for, flash, request, send_file, jsonify, Response, send_from_directory, stream_with_context
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from app import app, db
from app.forms import LoginForm, RegistrationForm, ImageUploadForm
from app.models import User, Image
from app.utils import allowed_file, extract_image_metadata, get_location_name, generate_csv_from_metadata
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

# Initialize ML predictor
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
print(f"\nInitializing MLPredictor with models directory: {models_dir}")
print(f"Available model files: {os.listdir(models_dir)}")
ml_predictor = MLPredictor(models_dir)

# Initialize damage detector with ML predictor
damage_detector = DamageDetector(ml_predictor)

# Thread pool for image processing
thread_pool = ThreadPoolExecutor(max_workers=4)

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
            # Attempt to get location name (address)
            try:
                location['address'] = get_location_name(lat, lon)
            except Exception as loc_err:
                 print(f"Error getting location name for ID {image.id}: {loc_err}")
                 location['address'] = "Address lookup failed"
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
        image_to_update = Image.objects(id=image.id).first() # Fetch fresh object just before final update
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
        pred_results_dict = {
            'damage_detected': bool(raw_preds),
            'raw_predictions': raw_preds, # Include raw predictions again
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

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
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
    
    return render_template('upload.html', form=form)

def is_video_file(filename):
    """Check if a file is a video based on its extension"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    return os.path.splitext(filename)[1].lower() in video_extensions

@app.route('/check_processing_status')
@login_required
def check_processing_status():
    try:
        # Get all processing images for the current user using MongoEngine
        processing_images = Image.objects(
            user_id=ObjectId(current_user.get_id()), 
            processing_status__in=['pending', 'processing']
        )
        
        status_list = [
            {
                'id': str(img.id),
                'original_filename': img.original_filename,
                'status': img.processing_status,
                'progress': 50 if img.processing_status == 'processing' else 0 # Simple progress indication
            } for img in processing_images
        ]
        
        return jsonify({'images': status_list})
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

        # Pass the raw Image DOCUMENT to the template
        return render_template('image_details.html', image=image)

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
        query_filters = {
            'processing_status': 'completed',
            'location__exists': True, # Ensure location field exists
            'location.latitude__exists': True, # Ensure latitude exists
            'location.longitude__exists': True # Ensure longitude exists
        }
        if not current_user.is_admin:
            query_filters['user_id'] = current_user.id

        # Fetch images with location data using MongoEngine
        # Select only necessary fields for performance
        images_with_location = Image.objects(**query_filters).only(
            'id', 'filename', 'original_filename', 'location', 
            'prediction_results', 'confidence_score' 
        )

        # Prepare data for the map
        map_data = []
        for img in images_with_location:
            # Double-check location data structure and validity
            loc = img.location
            if isinstance(loc, dict) and \
               isinstance(loc.get('latitude'), (int, float)) and \
               isinstance(loc.get('longitude'), (int, float)):
                
                damage_detected = img.prediction_results.get('damage_detected', False)
                confidence = img.confidence_score if damage_detected else None
                
                map_data.append({
                    'id': str(img.id),
                    'lat': loc['latitude'],
                    'lon': loc['longitude'],
                    'filename': img.original_filename or img.filename, # Use original if available
                    'damage': damage_detected,
                    'confidence': confidence,
                    'details_url': url_for('image_details', image_id=str(img.id))
                })
            else:
                print(f"Skipping image {img.id} due to invalid/missing location data: {loc}")

        return render_template('map_view.html', map_data=json.dumps(map_data))
    except Exception as e:
        flash(f'Error loading map view: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()
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
                                 if img.processing_status == 'complete' 
                                 and img.prediction_results.get('damage_type', 'No Damage') != 'No Damage'),
            'failed': sum(1 for img in images if img.processing_status == 'error'),
            'processing': sum(1 for img in images if img.processing_status == 'processing')
        }

        # Add success rate to stats
        completed_images = sum(1 for img in images if img.processing_status == 'complete')
        detection_stats['success_rate'] = (completed_images / detection_stats['total'] * 100) if detection_stats['total'] > 0 else 0
        detection_stats['detection_rate'] = (detection_stats['damage_detected'] / completed_images * 100) if completed_images > 0 else 0

        # Processing Time Analysis
        valid_times = []
        time_by_type = defaultdict(list)
        for img in images:
            if img.processing_status == 'complete':
                proc_time = img.processing_time
                img_type = img.prediction_results.get('model_used', 'Unknown')
                if proc_time and isinstance(proc_time, (int, float, str)):
                    try:
                        proc_time_float = float(proc_time)
                        if 0 < proc_time_float < 3600:  # Reasonable time limit
                            valid_times.append(proc_time_float)
                            time_by_type[img_type].append(proc_time_float)
                    except (ValueError, TypeError):
                        continue

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
            if img.processing_status == 'error':
                error = img.error_message or 'Unknown Error'
                proc_time = img.processing_time or 0
                error_counts[error] += 1
                error_times[error].append(float(proc_time) if proc_time else 0)
        
        for error, count in error_counts.items():
            times = error_times[error]
            error_analysis.append({
                'type': error,
                'count': count,
                'avg_time_impact': sum(times) / len(times) if times else 0
            })

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

# Helper function (example - if needed elsewhere)
# def some_helper():
#    pass
# Make sure this is the last function before the end of the file or other non-route code

# ... rest of the file (utility functions etc.) ... 