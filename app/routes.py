import os
import datetime
from flask import render_template, redirect, url_for, flash, request, send_file, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from app import app, db
from app.forms import LoginForm, RegistrationForm, ImageUploadForm
from app.models import User, Image
from app.utils import extract_image_metadata, generate_csv_from_metadata, get_location_name
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.get_by_username(form.username.data)
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
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=form.password.data,
            is_admin=False
        )
        user.save()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    # Get search term
    search_term = request.args.get('search', '').strip()
    
    # Base query
    query = {}
    
    # Filter by user if not admin
    if not current_user.is_admin:
        user_id_str = current_user.get_id()
        try:
            user_id_obj = ObjectId(user_id_str)
            query['$or'] = [
                {'user_id': user_id_str},
                {'user_id': user_id_obj}
            ]
        except Exception:
             # Fallback for invalid ID - query only string
             query['user_id'] = user_id_str

    # Add filename search condition if term exists
    if search_term:
        # Use regex for case-insensitive partial match
        # Need to escape regex special characters in the search term
        escaped_term = re.escape(search_term)
        regex_query = {'$regex': escaped_term, '$options': 'i'}
        
        # If query already has user filter ($or), add search to it
        if '$or' in query:
            # Add search term condition to BOTH parts of the $or for user filter
            query['$and'] = [
                {'$or': query.pop('$or')}, # Keep existing user filter
                {'filename': regex_query}  # Add filename filter
            ]
        else:
            # If only searching all images (admin) or failed user ID
            query['filename'] = regex_query
            
    print(f"[DEBUG DASHBOARD] Final Query: {query}") # DEBUG

    # Fetch images based on final query
    user_images = list(db.images.find(query).sort('upload_time', -1))
        
    return render_template('dashboard.html', images=user_images)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = ImageUploadForm()
    if form.validate_on_submit():
        uploaded_count = 0
        failed_count = 0
        
        # Loop through all uploaded files
        for image_file in form.images.data:
            try:
                original_filename = secure_filename(image_file.filename)
                file_ext = os.path.splitext(original_filename)[1].lower()
                unique_base = uuid.uuid4().hex

                # Ensure upload directory exists (redundant inside loop, but safe)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

                filename_to_store = f"{unique_base}{file_ext}"
                final_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_to_store)
                metadata_source_path = final_file_path # Default: extract from final file

                # Handle HEIC conversion
                if file_ext in ['.heic', '.heif']:
                    # Save HEIC temporarily for conversion
                    temp_heic_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_base}_temp{file_ext}")
                    jpeg_filename = f"{unique_base}.jpg"
                    final_file_path = os.path.join(app.config['UPLOAD_FOLDER'], jpeg_filename) # Final path is JPEG
                    filename_to_store = jpeg_filename # Store JPEG name in DB
                    
                    # Wrap HEIC processing in its own try block
                    try:
                        image_file.save(temp_heic_path)
                        
                        # Convert HEIC to JPEG using sips FIRST
                        sips_cmd = ["sips", "-s", "format", "jpeg", temp_heic_path, "--out", final_file_path]
                        result = subprocess.run(sips_cmd, capture_output=True, text=True, check=True) # Add check=True
                        
                        print(f"[INFO] Successfully converted HEIC to JPEG using sips: {final_file_path}")
                        metadata_source_path = final_file_path # Set metadata source to the NEW JPEG file
                        os.remove(temp_heic_path) # Clean up temporary HEIC *after* conversion
                        
                    except (subprocess.CalledProcessError, Exception) as heic_err:
                        # Handle HEIC specific errors (conversion failed)
                        flash(f'Error processing HEIC file {original_filename}: {heic_err}. Skipping this file.', 'danger')
                        if os.path.exists(temp_heic_path):
                            try: os.remove(temp_heic_path)
                            except OSError: pass
                        failed_count += 1
                        continue # Skip to the next file
                else:
                    # For non-HEIC files, save directly to the final path
                    image_file.save(final_file_path)
                    metadata_source_path = final_file_path # Extract from the original file

                # *** Extract metadata (now always from a JPEG or other Pillow-compatible file) ***
                print(f"[DEBUG UPLOAD] Extracting metadata from: {metadata_source_path}") # DEBUG
                metadata = extract_image_metadata(metadata_source_path)
                print(f"[DEBUG UPLOAD] Extracted metadata: {metadata}") # DEBUG
                
                # *** Perform Reverse Geocoding and add to metadata ***
                if metadata and not metadata.get('error') and metadata.get('latitude') and metadata.get('longitude'):
                    location_name = get_location_name(metadata.get('latitude'), metadata.get('longitude'))
                    if location_name:
                        metadata['location_name'] = location_name
                        print(f"[DEBUG UPLOAD] Added location_name: {location_name}") # DEBUG
                    else:
                        print("[DEBUG UPLOAD] Geocoding did not return a location name.") # DEBUG
                
                # Save image info to database
                if metadata is not None: # Check if metadata extraction failed
                    image_db_entry = Image(
                        filename=filename_to_store, # Use final filename (original or converted)
                        user_id=current_user.get_id(),
                        metadata=metadata,
                        upload_time=datetime.datetime.now()
                    )
                    image_db_entry.save()
                    uploaded_count += 1
                else:
                    flash(f'Failed to extract metadata for {original_filename}. Skipping database save.', 'warning')
                    failed_count += 1
                    # Optionally: delete the saved file if metadata extraction failed?
                    # if os.path.exists(final_file_path):
                    #     try: os.remove(final_file_path)
                    #     except OSError: pass 

            except Exception as general_err:
                 # Catch any unexpected errors during the processing of a single file
                 flash(f'An unexpected error occurred processing file {original_filename}: {general_err}. Skipping this file.', 'danger')
                 failed_count += 1
                 continue # Skip to next file
        
        # Flash summary message after processing all files
        if uploaded_count > 0:
             flash(f'{uploaded_count} image(s) uploaded successfully.', 'success')
        if failed_count > 0:
             flash(f'{failed_count} image(s) failed to upload or process.', 'danger')
             
        return redirect(url_for('dashboard'))
    
    return render_template('upload.html', form=form)

@app.route('/image/<image_id>')
@login_required
def image_details(image_id):
    image_data = Image.get_by_id(image_id)
    
    if not image_data or str(image_data['user_id']) != current_user.get_id():
        flash('Image not found or you do not have permission to view it.', 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('image_details.html', image=image_data)

@app.route('/image/delete/<image_id>', methods=['POST'])
@login_required
def delete_image(image_id):
    image_data = Image.get_by_id(image_id)

    # Verify image exists and belongs to the current user
    if not image_data or str(image_data['user_id']) != current_user.get_id():
        flash('Image not found or you do not have permission to delete it.', 'danger')
        return redirect(url_for('dashboard'))

    # Attempt to delete the image file
    filename = image_data['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_deleted = False
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            file_deleted = True
        else:
            # If file doesn't exist, still proceed to delete DB record
            file_deleted = True 
            flash(f'Image file {filename} not found, but removing database record.', 'warning')
    except OSError as e:
        flash(f'Error deleting image file: {e}', 'danger')
        # Optionally, decide if you want to proceed with DB deletion even if file deletion fails
        # return redirect(url_for('dashboard')) 

    # Attempt to delete the database record
    db_deleted = Image.delete_by_id(image_id)

    if db_deleted and file_deleted:
        flash('Image deleted successfully.', 'success')
    elif db_deleted and not file_deleted:
         # This case might happen if the file deletion failed but DB deletion succeeded
        flash('Image record deleted from database, but failed to delete the file.', 'warning')
    elif not db_deleted:
        flash('Failed to delete image record from the database.', 'danger')
        # Consider if you need to restore the file if it was deleted but DB op failed

    return redirect(url_for('dashboard'))

@app.route('/image/view/<filename>')
@login_required
def view_image(filename):
    # TODO: Add security check to verify user has access to this image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path)

@app.route('/export/csv')
@login_required
def export_csv():
    if current_user.is_admin:
        user_images = list(db.images.find())
        export_filename = f"all_metadata_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        user_images = Image.get_by_user_id(current_user.get_id())
        export_filename = f"metadata_export_{current_user.get_id()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Extract metadata from each image
    metadata_list = [img.get('metadata', {}) for img in user_images]
    
    # Generate CSV file
    csv_path = os.path.join(
        app.config['UPLOAD_FOLDER'], 
        export_filename
    )
    
    file_path = generate_csv_from_metadata(metadata_list, csv_path)
    
    return send_file(file_path, as_attachment=True)

@app.route('/map')
@login_required
def map_view():
    if current_user.is_admin:
        user_images = list(db.images.find())
    else:
        user_images = Image.get_by_user_id(current_user.get_id())
    
    # Filter images that have GPS coordinates
    geo_images = []
    for img in user_images:
        metadata = img.get('metadata', {})
        if 'latitude' in metadata and 'longitude' in metadata:
            geo_images.append({
                'id': str(img['_id']),
                'filename': img['filename'],
                'latitude': metadata['latitude'],
                'longitude': metadata['longitude'],
                'thumbnail': url_for('view_image', filename=img['filename'])
            })
    
    return render_template('map.html', geo_images=json.dumps(geo_images))

@app.route('/charts_data')
@login_required
def charts_data():
    """Endpoint to provide aggregated data for charts using efficient database queries."""
    print("[DEBUG CHARTS] /charts_data endpoint hit (Optimized Version)")

    # --- Determine Match Query based on User Role ---
    match_query = {}
    user_id_str = current_user.get_id()
    is_admin = current_user.is_admin

    if not is_admin:
        print(f"[DEBUG CHARTS] User is NOT admin. User ID (str): {user_id_str}")
        try:
            # Convert string ID to ObjectId for reliable matching
            user_id_obj = ObjectId(user_id_str)
            print(f"[DEBUG CHARTS] User ID (ObjectId): {user_id_obj}")
            match_query = {
                '$match': { 'user_id': user_id_obj } # Match directly on ObjectId
            }
            print(f"[DEBUG CHARTS] Constructed non-admin match query: {match_query}")
        except Exception as e:
             print(f"[WARN CHARTS] Invalid ObjectId format for user {user_id_str}. Error: {e}. Cannot generate user-specific charts.")
             # Return empty data structure if ID is invalid
             return jsonify({
                 'uploadsPerDay': {'labels': [], 'data': []},
                 'gpsDistribution': {'labels': [], 'data': []},
                 'formatDistribution': {'labels': [], 'data': []},
                 'locationClusters': {'labels': [], 'data': []},
                 'imagesPerUser': None # Keep consistent structure
             })
    else:
        print("[DEBUG CHARTS] User IS admin. Initial query will include all images.")
        match_query = {'$match': {}} # Start with an empty match for admin to include all

    # --- Define Aggregation Pipelines ---

    # Pipeline to get daily counts, GPS status, and format
    daily_and_format_pipeline = [match_query] # Start with user filter if needed
    daily_and_format_pipeline.extend([
        {
            '$project': {
                '_id': 0, # Exclude the default _id
                'upload_date_only': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$upload_time'}},
                'has_gps': {
                    '$cond': [
                        {'$and': [
                            {'$ne': ['$metadata.latitude', None]},
                            {'$ne': ['$metadata.longitude', None]},
                            {'$ne': ['$metadata.latitude', '']}, # Handle empty strings if they occur
                            {'$ne': ['$metadata.longitude', '']}
                        ]},
                        True, False
                    ]
                },
                'file_format': {
                    '$ifNull': ['$metadata.format', 'Unknown'] # Use stored format, default if missing
                }
            }
        },
        {
            '$group': {
                '_id': '$upload_date_only', # Group by date string
                'total_uploads': {'$sum': 1},
                'gps_count': {'$sum': {'$cond': ['$has_gps', 1, 0]}},
                # Count formats directly
                'formats': {'$push': '$file_format'}
            }
        },
        {
            '$sort': {'_id': 1} # Sort by date ascending
        }
    ])
    print(f"[DEBUG CHARTS] Daily & Format Pipeline: {daily_and_format_pipeline}")

    # Pipeline for Location Clusters (using stored location_name)
    location_pipeline = [match_query] # Start with user filter
    location_pipeline.extend([
        {
            '$match': { # Only include images with a valid location name
                'metadata.location_name': {'$exists': True, '$ne': None, '$ne': ''}
            }
        },
        {
            '$group': {
                '_id': '$metadata.location_name', # Group by the stored location name
                'count': {'$sum': 1}
            }
        },
        {'$sort': {'count': -1}}, # Sort by count descending
        {'$limit': 20} # Limit to top 20 locations
    ])
    print(f"[DEBUG CHARTS] Location Pipeline: {location_pipeline}")

    # Pipeline for Images per User (Admin only)
    admin_user_pipeline = []
    if is_admin:
        admin_user_pipeline.extend([
            # No initial match needed here, matches all users
            {
                '$group': {
                    '_id': '$user_id', # Group by user_id stored in image doc
                    'image_count': {'$sum': 1}
                }
            },
            { # Join with the users collection to get usernames
                '$lookup': {
                    'from': 'users', # The name of the users collection
                    'localField': '_id', # The user_id from the images collection
                    'foreignField': '_id', # The _id from the users collection
                    'as': 'user_info'
                }
            },
            { # Deconstruct the user_info array (should only be one match)
                '$unwind': {
                    'path': '$user_info',
                    'preserveNullAndEmptyArrays': True # Keep users even if not found in users collection
                }
            },
            { # Shape the output
                '$project': {
                    '_id': 0, # Exclude the default _id
                    'username': {
                        '$ifNull': ['$user_info.username', {'$concat': ['Unknown User (ID: ', {'$toString': '$_id'}, ')']}]
                    },
                    'image_count': 1
                }
            },
            {'$sort': {'image_count': -1}} # Sort by image count descending
        ])
        print(f"[DEBUG CHARTS] Admin User Pipeline: {admin_user_pipeline}")

    # --- Execute Aggregations ---
    try:
        print("[DEBUG CHARTS] Executing daily/format pipeline...")
        daily_format_results = list(db.images.aggregate(daily_and_format_pipeline))
        print(f"[DEBUG CHARTS] Daily/Format results count: {len(daily_format_results)}")

        print("[DEBUG CHARTS] Executing location pipeline...")
        location_results = list(db.images.aggregate(location_pipeline))
        print(f"[DEBUG CHARTS] Location results count: {len(location_results)}")

        admin_user_results = []
        if is_admin:
            print("[DEBUG CHARTS] Executing admin user pipeline...")
            admin_user_results = list(db.images.aggregate(admin_user_pipeline))
            print(f"[DEBUG CHARTS] Admin user results count: {len(admin_user_results)}")

    except Exception as agg_error:
        print(f"[ERROR CHARTS] Aggregation failed: {agg_error}")
        # Return empty structure on error
        return jsonify({
            'uploadsPerDay': {'labels': [], 'data': []},
            'gpsDistribution': {'labels': [], 'data': []},
            'formatDistribution': {'labels': [], 'data': []},
            'locationClusters': {'labels': [], 'data': []},
            'imagesPerUser': {'labels': [], 'data': []} if is_admin else None
        })

    # --- Process Results ---

    # Uploads per Day
    uploads_daily_labels = [res['_id'] for res in daily_format_results]
    uploads_daily_data = [res['total_uploads'] for res in daily_format_results]
    print(f"[DEBUG CHARTS] Processed daily uploads (labels: {len(uploads_daily_labels)}, data: {len(uploads_daily_data)})")

    # GPS Distribution
    total_images = sum(res['total_uploads'] for res in daily_format_results)
    total_gps = sum(res['gps_count'] for res in daily_format_results)
    total_no_gps = total_images - total_gps
    print(f"[DEBUG CHARTS] Total Images: {total_images}, With GPS: {total_gps}, Without GPS: {total_no_gps}")

    # Format Distribution
    format_counts = Counter()
    for res in daily_format_results:
        format_counts.update(res.get('formats', [])) # Aggregate format counts
    format_labels = list(format_counts.keys())
    format_data = list(format_counts.values())
    print(f"[DEBUG CHARTS] Format Counts: {format_counts}")

    # Location Clusters
    location_labels = [res['_id'] for res in location_results]
    location_data = [res['count'] for res in location_results]
    print(f"[DEBUG CHARTS] Location Clusters (labels: {len(location_labels)}, data: {len(location_data)})")

    # Images per User (Admin)
    user_labels = []
    user_data = []
    if is_admin:
        user_labels = [res['username'] for res in admin_user_results]
        user_data = [res['image_count'] for res in admin_user_results]
        print(f"[DEBUG CHARTS] Images Per User (labels: {len(user_labels)}, data: {len(user_data)})")


    # --- Construct Final JSON Response ---
    chart_data = {
        'uploadsPerDay': {
            'labels': uploads_daily_labels,
            'data': uploads_daily_data,
        },
        'gpsDistribution': {
            'labels': ['With GPS', 'Without GPS'],
            'data': [total_gps, total_no_gps] if total_images > 0 else []
        },
        'formatDistribution': {
            'labels': format_labels,
            'data': format_data
        },
        'locationClusters': {
            'labels': location_labels,
            'data': location_data
        },
        'imagesPerUser': None
    }

    if is_admin:
        chart_data['imagesPerUser'] = {
            'labels': user_labels,
            'data': user_data
        }

    print(f"[DEBUG CHARTS] Final chart_data payload: {json.dumps(chart_data, indent=2)}") # Pretty print for debugging
    return jsonify(chart_data)

# Helper function (example - if needed elsewhere)
# def some_helper():
#    pass
# Make sure this is the last function before the end of the file or other non-route code

# ... rest of the file (utility functions etc.) ... 