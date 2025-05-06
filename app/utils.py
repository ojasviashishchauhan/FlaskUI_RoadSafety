import os
import pandas as pd
from PIL import Image as PILImage
from PIL.ExifTags import TAGS, GPSTAGS
from PIL.TiffImagePlugin import IFDRational
import datetime
import uuid
import collections.abc
from pillow_heif import register_heif_opener
import io
import exifread # Import exifread
from exifread.utils import Ratio # Import Ratio for type checking
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from app import app
import subprocess # Add subprocess for ffprobe
import json # Add json for parsing ffprobe output

# Initialize geolocator globally but cautiously (consider thread safety if scaling heavily)
# Using a specific user_agent is required by Nominatim's policy
geolocator = Nominatim(user_agent="flask_image_mapper_v1")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic', 'heif'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    video_extensions = {'mp4', 'avi', 'mov', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def _sanitize_for_bson(data):
    """Recursively sanitize data to ensure BSON compatibility."""
    if isinstance(data, collections.abc.Mapping):
        # Handle dictionaries (and other mapping types)
        # Convert all keys to strings to ensure BSON compatibility
        return {str(key): _sanitize_for_bson(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        # Handle lists and tuples
        return [_sanitize_for_bson(item) for item in data]
    elif isinstance(data, IFDRational):
        # Convert IFDRational to float, handling zero denominator
        if data.denominator == 0:
            return None # Or float('nan'), or 0.0, depending on desired handling
        else:
            return float(data)
    elif isinstance(data, Ratio): # Handle exifread Ratio
        if data.den == 0:
            return None
        else:
            return float(data.num) / float(data.den)
    elif isinstance(data, bytes):
        # Decode bytes to string
        try:
            return data.decode('utf-8', errors='replace')
        except Exception:
            return str(data) # Fallback to string representation
    elif isinstance(data, (int, float, str, bool, datetime.datetime, type(None))):
        # Allow standard BSON types
        return data
    else:
        # Convert any other unsupported types to string as a fallback
        return str(data)

def _convert_exifread_gps_to_degrees(value):
    """Converts GPS data from exifread format (list of Ratio) to degrees."""
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        d = float(value[0].num) / float(value[0].den)
        m = float(value[1].num) / float(value[1].den)
        s = float(value[2].num) / float(value[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    except (ZeroDivisionError, TypeError, IndexError):
        return None

def extract_image_metadata(image_path):
    """
    Extract metadata from an image file using Pillow or exifread
    
    Returns a dictionary with metadata information
    """
    print(f"[DEBUG] Starting metadata extraction for: {image_path}") # DEBUG
    try:
        metadata = {}
        
        # Check if this is a HEIC/HEIF file based on file extension
        is_heic = image_path.lower().endswith(('.heic', '.heif'))
        
        # For HEIC files, skip Pillow opening and use exifread directly
        if is_heic:
            print(f"[DEBUG] Detected HEIC/HEIF file by extension, using exifread directly") # DEBUG
            # Set basic metadata (we can't get this from exifread)
            metadata['filename'] = os.path.basename(image_path)
            
            try:
                with open(image_path, 'rb') as f:
                    tags = exifread.process_file(f)
                
                print(f"[DEBUG] ExifRead found {len(tags)} tags")
                
                # Store all tags for reference
                exif_data = {str(tag): str(tags[tag]) for tag in tags}
                
                # Extract specific tags we're interested in
                
                # Get image dimensions if available
                if 'EXIF ExifImageWidth' in tags and 'EXIF ExifImageLength' in tags:
                    try:
                        metadata['width'] = int(str(tags['EXIF ExifImageWidth']))
                        metadata['height'] = int(str(tags['EXIF ExifImageLength']))
                    except (ValueError, TypeError):
                        print("[WARN] Failed to parse image dimensions from EXIF")
                
                # Extract camera model information
                if 'Image Make' in tags:
                    metadata['camera_make'] = str(tags['Image Make']).strip()
                if 'Image Model' in tags:
                    metadata['camera_model'] = str(tags['Image Model']).strip()
                    print(f"[DEBUG] Found camera model: {metadata['camera_model']}")
                
                # Extract GPS data
                lat_tag = tags.get('GPS GPSLatitude')
                lat_ref_tag = tags.get('GPS GPSLatitudeRef')
                lon_tag = tags.get('GPS GPSLongitude')
                lon_ref_tag = tags.get('GPS GPSLongitudeRef')
                
                if lat_tag and lat_ref_tag:
                    lat = _convert_exifread_gps_to_degrees(lat_tag.values)
                    if lat is not None:
                        if str(lat_ref_tag) == 'S':
                            lat = -lat
                        metadata['latitude'] = lat
                        print(f"[DEBUG] Extracted Latitude: {lat}") # DEBUG
                    else:
                        print("[WARN] Could not convert latitude values")
                
                if lon_tag and lon_ref_tag:
                    lon = _convert_exifread_gps_to_degrees(lon_tag.values)
                    if lon is not None:
                        if str(lon_ref_tag) == 'W':
                            lon = -lon
                        metadata['longitude'] = lon
                        print(f"[DEBUG] Extracted Longitude: {lon}") # DEBUG
                    else:
                        print("[WARN] Could not convert longitude values")
                
                # Extract date taken
                if 'EXIF DateTimeOriginal' in tags:
                    metadata['date_taken'] = str(tags['EXIF DateTimeOriginal'])
                
                # Add the exif_data to metadata
                metadata['exif'] = exif_data
                
                # Add format info
                metadata['format'] = 'HEIF'
                metadata['mode'] = 'Unknown'  # Can't determine mode without opening
                
                # Sanitize and return metadata here for HEIC files
                print(f"[DEBUG] HEIC metadata before sanitization: {metadata}") # DEBUG
                # Final sanitization pass on the whole dictionary before returning
                sanitized_metadata = _sanitize_for_bson(metadata)
                print(f"[DEBUG] HEIC metadata after sanitization: {sanitized_metadata}") # DEBUG
                return sanitized_metadata
                
            except Exception as e:
                print(f"[ERROR] Failed to extract HEIC metadata with exifread: {e}")
                # Return basic metadata with error
                metadata['error'] = str(e)
                return _sanitize_for_bson(metadata)
        else:
            # For non-HEIC files, use Pillow as before
            image = PILImage.open(image_path)
            print(f"[DEBUG] Image opened successfully: {image.format} {image.size} {image.mode}") # DEBUG
            
            # Basic metadata
            metadata['filename'] = os.path.basename(image_path)
            metadata['width'], metadata['height'] = image.size
            metadata['format'] = image.format
            metadata['mode'] = image.mode
            
            # Get EXIF data
            exif_data = {}
            if image.format == 'HEIF':
                print("[DEBUG] Image format is HEIF, using exifread library") # DEBUG
                try:
                    with open(image_path, 'rb') as f:
                        tags = exifread.process_file(f)
                    
                    print(f"[DEBUG HEIF ExifRead] Found {len(tags)} tags.") # DEBUG
                    # Store raw tags (after sanitization) for display if needed
                    exif_data = {tag: tags[tag] for tag in tags}
                    
                    # Extract camera model information
                    if 'Image Make' in tags:
                        metadata['camera_make'] = str(tags['Image Make']).strip()
                    if 'Image Model' in tags:
                        metadata['camera_model'] = str(tags['Image Model']).strip()
                        print(f"[DEBUG] Found camera model: {metadata['camera_model']}")
                    
                    # Extract GPS data using exifread tags
                    lat_tag = tags.get('GPS GPSLatitude')
                    lat_ref_tag = tags.get('GPS GPSLatitudeRef')
                    lon_tag = tags.get('GPS GPSLongitude')
                    lon_ref_tag = tags.get('GPS GPSLongitudeRef')
                    
                    if lat_tag and lat_ref_tag:
                        lat = _convert_exifread_gps_to_degrees(lat_tag.values)
                        if lat is not None:
                            if lat_ref_tag.values == 'S':
                                lat = -lat
                            metadata['latitude'] = lat
                            print(f"[DEBUG HEIF ExifRead] Calculated Latitude: {lat}") # DEBUG
                        else:
                            print("[WARN HEIF ExifRead] Could not convert latitude values.")
                    else:
                         print("[DEBUG HEIF ExifRead] Latitude tags not found.")

                    if lon_tag and lon_ref_tag:
                        lon = _convert_exifread_gps_to_degrees(lon_tag.values)
                        if lon is not None:
                            if lon_ref_tag.values == 'W':
                                lon = -lon
                            metadata['longitude'] = lon
                            print(f"[DEBUG HEIF ExifRead] Calculated Longitude: {lon}") # DEBUG
                        else:
                            print("[WARN HEIF ExifRead] Could not convert longitude values.")
                    else:
                         print("[DEBUG HEIF ExifRead] Longitude tags not found.")
                     
                    # Extract Date Taken
                    date_tag = tags.get('EXIF DateTimeOriginal')
                    if date_tag:
                        metadata['date_taken'] = str(date_tag.values)
                        print(f"[DEBUG HEIF ExifRead] Date Taken: {metadata['date_taken']}") # DEBUG
                    else:
                        print("[DEBUG HEIF ExifRead] DateTimeOriginal tag not found.")
                except Exception as e:
                    print(f"[ERROR] Failed to process HEIF with exifread: {e}")
            else:
                # Use Pillow's _getexif for other formats (e.g., JPEG)
                print(f"[DEBUG] Image format is {image.format}, using image._getexif()") # DEBUG
                raw_exif_dict = None
                if hasattr(image, '_getexif'):
                    raw_exif_dict = image._getexif()
                    if raw_exif_dict:
                        print(f"[DEBUG] Raw EXIF data found (length: {len(raw_exif_dict)})") # DEBUG
                        
                        # Extract camera model information
                        for tag_id, value in raw_exif_dict.items():
                            tag_name = TAGS.get(tag_id, str(tag_id))
                            if tag_name == 'Make':
                                metadata['camera_make'] = str(value).strip()
                            elif tag_name == 'Model':
                                metadata['camera_model'] = str(value).strip()
                                print(f"[DEBUG] Found camera model: {metadata['camera_model']}")
                            
                            # Store all EXIF data
                            if isinstance(tag_name, str):
                                exif_data[tag_name] = value
                        
                        # Handle GPS data
                        if 'GPSInfo' in exif_data:
                            gps_data = {}
                            for gps_tag_id, gps_value in exif_data['GPSInfo'].items():
                                gps_tag_name = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                                gps_data[gps_tag_name] = gps_value
                            
                            if all(k in gps_data for k in ['GPSLatitude', 'GPSLatitudeRef', 'GPSLongitude', 'GPSLongitudeRef']):
                                try:
                                    lat = _convert_to_degrees(gps_data['GPSLatitude'])
                                    if gps_data['GPSLatitudeRef'] == 'S':
                                        lat = -lat
                                    lon = _convert_to_degrees(gps_data['GPSLongitude'])
                                    if gps_data['GPSLongitudeRef'] == 'W':
                                        lon = -lon
                                    metadata['latitude'] = lat
                                    metadata['longitude'] = lon
                                except Exception as e:
                                    print(f"[ERROR] Failed to process GPS data: {e}")
                    else:
                        print(f"[DEBUG] image._getexif() returned None or empty.") # DEBUG
                else:
                    print(f"[DEBUG] image object has no _getexif attribute.") # DEBUG
            
            # Add the processed EXIF data to metadata
            metadata['exif'] = _sanitize_for_bson(exif_data)
            
            # Final sanitization pass on the whole dictionary before returning
            return _sanitize_for_bson(metadata)
            
    except Exception as e:
        print(f"Error extracting metadata for {image_path}: {e}")
        return {'error': str(e)} # Return error in metadata

def _convert_to_degrees(value):
    """
    Helper function to convert GPS coordinates from Pillow EXIF format (tuple of IFDRational) to decimal degrees
    """
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    try:
        # Handle IFDRational if present
        d = float(value[0]) if not isinstance(value[0], IFDRational) else float(value[0])
        m = float(value[1]) if not isinstance(value[1], IFDRational) else float(value[1])
        s = float(value[2]) if not isinstance(value[2], IFDRational) else float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except (ZeroDivisionError, TypeError, IndexError):
         return None

def generate_csv_from_metadata(metadata_list, output_path=None):
    """
    Generate a CSV file from a list of metadata dictionaries
    
    Returns the path to the generated CSV file
    """
    # Flatten the metadata for CSV format
    flattened_data = []
    
    for metadata in metadata_list:
        flat_metadata = {
            'filename': metadata.get('filename', ''),
            'width': metadata.get('width', ''),
            'height': metadata.get('height', ''),
            'format': metadata.get('format', ''),
            'date_taken': metadata.get('date_taken', ''),
            'latitude': metadata.get('latitude', ''),
            'longitude': metadata.get('longitude', '')
        }
        flattened_data.append(flat_metadata)
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(flattened_data)
    
    if not output_path:
        # Generate unique filename if not provided
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        output_path = f'image_metadata_{timestamp}_{unique_id}.csv'
    
    df.to_csv(output_path, index=False)
    return output_path 

def get_location_name(lat, lon):
    """
    Get location name from coordinates using Nominatim geocoder
    Args:
        lat: Latitude
        lon: Longitude
    Returns:
        String with formatted location information
    """
    if not lat or not lon:
        return "Unknown Location"

    try:
        # Use Nominatim to get location info with detailed address
        geolocator = Nominatim(user_agent="flask_road_damage_detector")
        location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, language='en')
        
        if location and location.raw.get('address'):
            address = location.raw['address']
            
            # Extract relevant address components
            road = address.get('road', '')
            suburb = address.get('suburb', '')
            city = address.get('city', address.get('town', address.get('village', '')))
            state = address.get('state', '')
            country = address.get('country', '')
            
            # Construct detailed address
            location_parts = []
            if road:
                location_parts.append(road)
            if suburb and suburb != road:
                location_parts.append(suburb)
            if city and city not in location_parts:
                location_parts.append(city)
            if state and state not in location_parts:
                location_parts.append(state)
            if country and country not in location_parts:
                location_parts.append(country)
            
            return ", ".join(location_parts)
        else:
            return "Coordinates found but no address information available"
    except GeocoderTimedOut:
        return "Location lookup timed out"
    except GeocoderServiceError:
        return "Location service error"
    except Exception as e:
        print(f"Error in get_location_name: {e}")
        return "Error determining location"

def get_location_details(lat, lon, max_retries=3, timeout=10):
    """
    Get structured location details from coordinates using Nominatim geocoder
    with retry logic and fallback options
    
    Args:
        lat: Latitude
        lon: Longitude
        max_retries: Maximum number of retries if geocoding fails
        timeout: Timeout in seconds for geocoding request
    
    Returns:
        Dictionary with structured location information including city, state, and full address
    """
    if not lat or not lon:
        return {
            "full_address": "Unknown Location",
            "city": "Unknown",
            "state": "Unknown",
            "country": "Unknown",
            "formatted_address": "Unknown Location"
        }

    # Try different geocoding services in order
    geocoding_services = [
        lambda: _get_location_from_nominatim(lat, lon, max_retries, timeout),
        lambda: _get_location_from_coordinates(lat, lon)  # Simple fallback using coordinates
    ]
    
    for service in geocoding_services:
        try:
            result = service()
            if result and result.get('city') != 'Unknown':
                # We found a good result, return it
                return result
        except Exception as e:
            print(f"Geocoding service error: {e}")
            continue
    
    # If all services fail, build a basic result using coordinates
    return {
        "full_address": f"Coordinates: {lat}, {lon}",
        "city": f"Location ({round(float(lat), 4)}, {round(float(lon), 4)})",
        "state": "Unknown",
        "country": "Unknown",
        "latitude": lat,
        "longitude": lon,
        "formatted_address": f"Coordinates: {lat}, {lon}"
    }

def _get_location_from_nominatim(lat, lon, max_retries=3, timeout=10):
    """
    Attempts to get location from Nominatim with retries
    """
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    import time
    
    for attempt in range(max_retries):
        try:
            # Use Nominatim to get detailed location info with increasing timeout
            current_timeout = timeout * (1 + attempt * 0.5)  # Increase timeout with each retry
            geolocator = Nominatim(user_agent=f"flask_road_damage_detector_retry_{attempt}", timeout=current_timeout)
            location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, language='en')
            
            if location and location.raw.get('address'):
                address = location.raw['address']
                
                # Extract address components
                city = address.get('city', address.get('town', address.get('village', 'Unknown')))
                state = address.get('state', 'Unknown')
                country = address.get('country', 'Unknown')
                postal_code = address.get('postcode', '')
                
                # Additional location details
                road = address.get('road', '')
                suburb = address.get('suburb', '')
                county = address.get('county', '')
                district = address.get('state_district', '')
                
                # Construct formatted address
                location_parts = []
                if road:
                    location_parts.append(road)
                if suburb and suburb not in location_parts:
                    location_parts.append(suburb)
                if city and city not in location_parts:
                    location_parts.append(city)
                if state:
                    location_parts.append(state)
                if country:
                    location_parts.append(country)
                
                # If city is still unknown but we have county, use that
                if city == 'Unknown' and county:
                    city = county
                
                # If city is still unknown but we have district, use that
                if city == 'Unknown' and district:
                    city = district
                
                formatted_address = ", ".join(location_parts)
                
                # Build structured response
                return {
                    "full_address": address,
                    "city": city,
                    "state": state,
                    "country": country,
                    "postal_code": postal_code,
                    "road": road,
                    "suburb": suburb,
                    "county": county,
                    "district": district,
                    "latitude": lat,
                    "longitude": lon,
                    "formatted_address": formatted_address,
                    "display_name": location.raw.get('display_name', formatted_address)
                }
            else:
                # Try again if no result was found
                print(f"No data from Nominatim on attempt {attempt+1}, retrying...")
                time.sleep(1)  # Brief pause before retry
                continue
                
        except GeocoderTimedOut:
            print(f"Geocoder timed out for coordinates: {lat}, {lon} (attempt {attempt+1})")
            time.sleep(2)  # Wait before retry
            continue
        except GeocoderServiceError:
            print(f"Geocoder service error for coordinates: {lat}, {lon} (attempt {attempt+1})")
            time.sleep(3)  # Longer wait for service errors
            continue
        except Exception as e:
            print(f"Error in geocoding on attempt {attempt+1}: {e}")
            time.sleep(1)
            continue
    
    # If we got here, all attempts failed
    return {
        "full_address": "No address found after retries",
        "city": "Unknown",
        "state": "Unknown",
        "country": "Unknown",
        "latitude": lat,
        "longitude": lon,
        "formatted_address": "Unable to determine location after multiple attempts"
    }

def _get_location_from_coordinates(lat, lon):
    """
    Simple fallback method that just formats the coordinates into a readable form
    """
    try:
        # Round coordinates for display
        lat_rounded = round(float(lat), 4)
        lon_rounded = round(float(lon), 4)
        
        return {
            "full_address": f"Coordinates: {lat_rounded}, {lon_rounded}",
            "city": f"Location ({lat_rounded}, {lon_rounded})",
            "state": "Unknown",
            "country": "Unknown",
            "latitude": lat,
            "longitude": lon,
            "formatted_address": f"Coordinates: {lat_rounded}, {lon_rounded}"
        }
    except Exception as e:
        print(f"Error in coordinate fallback: {e}")
        return None

# --- New function for Video Metadata --- 
def extract_video_metadata(video_path):
    """Extracts metadata from a video file using ffprobe (if available) and OpenCV."""
    metadata = {'error': None}
    
    # --- Try ffprobe first for detailed metadata including GPS --- 
    try:
        # Command to get metadata in JSON format, show streams
        command = [
            'ffprobe', 
            '-v', 'quiet',          # Suppress unnecessary logging
            '-print_format', 'json', # Output as JSON
            '-show_format',         # Include container format info
            '-show_streams',        # Include stream info (video, audio, data)
            video_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        ffprobe_data = json.loads(result.stdout)
        
        # Extract format information
        if 'format' in ffprobe_data:
            fmt_info = ffprobe_data['format']
            metadata['format_name'] = fmt_info.get('format_name')
            metadata['duration_seconds'] = float(fmt_info.get('duration', 0))
            metadata['size_bytes'] = int(fmt_info.get('size', 0))
            metadata['bit_rate'] = int(fmt_info.get('bit_rate', 0))
            
            # Extract format tags (often contains rotation, creation_time, sometimes GPS)
            if 'tags' in fmt_info:
                 metadata['format_tags'] = fmt_info['tags'] # Store all format tags
                 # Check common GPS tags in format metadata
                 if 'location' in fmt_info['tags']:
                     # Format is often like "+12.345-078.910/" - needs parsing
                     loc_str = fmt_info['tags']['location']
                     try:
                          # Simple regex assuming '+/-DD.DDD+/-DDD.DDD/' format
                          import re
                          match = re.match(r"([+-]\d+\.\d+)([+-]\d+\.\d+)", loc_str)
                          if match:
                               metadata['latitude'] = float(match.group(1))
                               metadata['longitude'] = float(match.group(2))
                     except Exception as loc_parse_err:
                          print(f"Warning: Could not parse GPS location tag: {loc_str} - Error: {loc_parse_err}")
                 elif 'creation_time' in fmt_info['tags']:
                      metadata['creation_time'] = fmt_info['tags']['creation_time']

        # Extract video stream information
        video_stream = next((s for s in ffprobe_data.get('streams', []) if s.get('codec_type') == 'video'), None)
        if video_stream:
            metadata['frame_width'] = video_stream.get('width')
            metadata['frame_height'] = video_stream.get('height')
            metadata['codec_name'] = video_stream.get('codec_name')
            metadata['profile'] = video_stream.get('profile')
            # Calculate FPS if possible
            avg_fps_str = video_stream.get('avg_frame_rate', '0/0')
            try:
                num, den = map(int, avg_fps_str.split('/'))
                if den > 0:
                    metadata['fps'] = round(num / den, 2)
                else: 
                    metadata['fps'] = None
            except ValueError:
                metadata['fps'] = None
            metadata['stream_tags'] = video_stream.get('tags', {}) # Store video stream tags
            # Check for rotation in video stream tags
            if 'rotate' in metadata['stream_tags']:
                 metadata['rotation'] = metadata['stream_tags']['rotate']
            # Check for creation time in video stream tags
            if 'creation_time' in metadata['stream_tags'] and 'creation_time' not in metadata:
                 metadata['creation_time'] = metadata['stream_tags']['creation_time']
        
        # Extract data stream information (look for GPS here too)
        gps_stream = next((s for s in ffprobe_data.get('streams', []) if s.get('codec_tag_string') == 'gpmd' or s.get('codec_tag_string') == 'rtmd'), None)
        if gps_stream and 'tags' in gps_stream:
             metadata['gps_stream_tags'] = gps_stream['tags']
             # Look for specific GPS coordinate tags within the data stream
             # Keys can vary wildly depending on camera/muxer (e.g., 'location', 'GPSCoordinates')
             if 'location' in gps_stream['tags'] and 'latitude' not in metadata:
                  loc_str = gps_stream['tags']['location']
                  try:
                       import re
                       match = re.match(r"([+-]\d+\.\d+)([+-]\d+\.\d+)", loc_str)
                       if match:
                           metadata['latitude'] = float(match.group(1))
                           metadata['longitude'] = float(match.group(2))
                  except Exception as loc_parse_err:
                       print(f"Warning: Could not parse GPS location tag from data stream: {loc_str} - Error: {loc_parse_err}")

        print(f"Successfully extracted video metadata using ffprobe for: {os.path.basename(video_path)}")

    except FileNotFoundError:
        msg = "ffprobe (part of FFmpeg) not found. Cannot extract detailed video metadata including GPS. Falling back to OpenCV."
        print(f"WARNING: {msg}")
        metadata['warning'] = msg # Add warning instead of error for fallback
    except subprocess.CalledProcessError as e:
        msg = f"ffprobe command failed for {os.path.basename(video_path)}: {e.stderr}"
        print(f"WARNING: {msg}")
        metadata['warning'] = msg
    except json.JSONDecodeError as e:
        msg = f"Failed to parse ffprobe JSON output for {os.path.basename(video_path)}: {e}"
        print(f"WARNING: {msg}")
        metadata['warning'] = msg
    except Exception as e:
        msg = f"Unexpected error during ffprobe processing for {os.path.basename(video_path)}: {e}"
        print(f"WARNING: {msg}")
        metadata['warning'] = msg
        
    # --- Fallback/Supplement with OpenCV for basic info if needed --- 
    # Especially useful if ffprobe failed or didn't provide dimensions/fps
    if 'frame_width' not in metadata or 'fps' not in metadata:
        print(f"Attempting OpenCV fallback for basic metadata: {os.path.basename(video_path)}")
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                # Don't overwrite ffprobe error if it exists
                if not metadata.get('error') and not metadata.get('warning'):
                     metadata['error'] = f"OpenCV fallback failed: Cannot open video file: {video_path}"
                raise IOError(metadata['error'] or "Cannot open video") # Raise exception to stop
                
            # Only add if missing from ffprobe data
            if 'frame_width' not in metadata:
                 metadata['frame_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if 'frame_height' not in metadata:
                 metadata['frame_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if 'fps' not in metadata:
                 fps = cap.get(cv2.CAP_PROP_FPS)
                 metadata['fps'] = round(fps, 2) if fps else None
            if 'frame_count' not in metadata:
                 frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                 metadata['frame_count'] = frame_count
            if 'duration_seconds' not in metadata and metadata.get('fps') and metadata.get('frame_count'):
                 duration = metadata['frame_count'] / metadata['fps']
                 metadata['duration_seconds'] = round(duration, 2)
                 td = datetime.timedelta(seconds=duration)
                 metadata['duration_formatted'] = str(td).split('.')[0]
            if 'codec_fourcc' not in metadata:
                 fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                 metadata['codec_fourcc'] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            print(f"Successfully supplemented video metadata with OpenCV for: {os.path.basename(video_path)}")
        except ImportError:
            msg = "OpenCV (cv2) is not installed. Cannot perform fallback metadata extraction."
            print(f"ERROR: {msg}")
            if not metadata.get('error') and not metadata.get('warning'): # Only set error if ffprobe didn't already fail
                 metadata['error'] = msg
        except Exception as e:
            # Don't overwrite ffprobe error/warning
            if not metadata.get('error') and not metadata.get('warning'):
                msg = f"Error during OpenCV fallback for {os.path.basename(video_path)}: {e}"
                print(f"ERROR: {msg}")
                metadata['error'] = str(e)
            if 'cap' in locals() and cap.isOpened():
                cap.release()

    # Final sanitization before returning
    return _sanitize_for_bson(metadata)
# --- End Video Metadata function --- 

# --- IoU Calculation Utility --- 
def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are expected in format [x1, y1, x2, y2].
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou
# --- End IoU Calculation --- 

# Function to reverse geocode coordinates to get location name
def get_location_name(lat, lon):
    # Implementation of the function
    pass 