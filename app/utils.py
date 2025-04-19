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

# Initialize geolocator globally but cautiously (consider thread safety if scaling heavily)
# Using a specific user_agent is required by Nominatim's policy
geolocator = Nominatim(user_agent="flask_image_mapper_v1")

def _sanitize_for_bson(data):
    """Recursively sanitize data to ensure BSON compatibility."""
    if isinstance(data, collections.abc.Mapping):
        # Handle dictionaries (and other mapping types)
        return {key: _sanitize_for_bson(value) for key, value in data.items()}
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
                        tags = exifread.process_file(f, stop_tag='DateTimeOriginal') # Optimization
                    
                    print(f"[DEBUG HEIF ExifRead] Found {len(tags)} tags.") # DEBUG
                    # Store raw tags (after sanitization) for display if needed
                    exif_data = {tag: tags[tag] for tag in tags} 
                    
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
                    else:
                         print(f"[DEBUG] image._getexif() returned None or empty.") # DEBUG
                else:
                     print(f"[DEBUG] image object has no _getexif attribute.") # DEBUG

                # Process the extracted EXIF dictionary (if any)
                if raw_exif_dict:
                    # Create a new dictionary to store EXIF data with string keys
                    string_keyed_exif_data = {}
                    
                    for tag_id, value in raw_exif_dict.items():
                        # Decode the tag ID into a string name, fallback to str(ID)
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        
                        # Use the known tag name (if available) for specific checks
                        tag = TAGS.get(tag_id) 
                        
                        # Handle GPS data specially (using Pillow's format)
                        if tag == 'GPSInfo':
                            print(f"[DEBUG] Found GPSInfo tag (ID: {tag_id})") # DEBUG
                            gps_info_dict = value
                            gps_data = {} # Used for lat/lon extraction
                            string_keyed_gps_data = {} # For storing in metadata['exif']
                            
                            if isinstance(gps_info_dict, dict):
                                for gps_tag_id, gps_value in gps_info_dict.items():
                                    # Get string key for GPS tag
                                    gps_tag_name = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                                    string_keyed_gps_data[gps_tag_name] = gps_value # Store with string key
                                    
                                    # Also populate gps_data for lat/lon calculation
                                    # (using the standard tag name if available)
                                    gps_tag = GPSTAGS.get(gps_tag_id)
                                    if gps_tag:
                                         gps_data[gps_tag] = gps_value
                            else:
                                print(f"[WARN] GPSInfo value was not a dictionary (Type: {type(gps_info_dict)}). Cannot parse GPS tags.")
                            
                            # Store the string-keyed GPS dict within the main string-keyed EXIF dict
                            string_keyed_exif_data[tag_name] = string_keyed_gps_data
                            
                            print(f"[DEBUG GPS] Parsed gps_data dictionary for lat/lon calc: {gps_data}")
                            
                            lat = None
                            lon = None
                            if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
                                lat = _convert_to_degrees(gps_data['GPSLatitude'])
                                if lat is not None:
                                    if gps_data['GPSLatitudeRef'] == 'S':
                                        lat = -lat
                                    metadata['latitude'] = lat
                                    print(f"[DEBUG] Calculated Latitude: {lat}") # DEBUG
                        
                            if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
                                lon = _convert_to_degrees(gps_data['GPSLongitude'])
                                if lon is not None:
                                    if gps_data['GPSLongitudeRef'] == 'W':
                                        lon = -lon
                                    metadata['longitude'] = lon
                                    print(f"[DEBUG] Calculated Longitude: {lon}") # DEBUG
                            else:
                                print("[WARN] Could not convert longitude values.")
                        else:
                            # For non-GPS tags, store directly with string key
                            string_keyed_exif_data[tag_name] = value 
                
                    # Add date taken if available (check by tag ID in raw dict)
                    dateTimeOriginalTagId = 36867 # EXIF DateTimeOriginal tag ID
                    if dateTimeOriginalTagId in raw_exif_dict:
                        # Ensure date_taken is also added to the string_keyed dict if needed for consistency
                        # Though it's primarily stored directly in metadata
                        date_str = str(raw_exif_dict[dateTimeOriginalTagId])
                        metadata['date_taken'] = date_str 
                        string_keyed_exif_data['DateTimeOriginal'] = date_str 
                
                    # Assign the dictionary with STRING keys to metadata['exif']
                    metadata['exif'] = string_keyed_exif_data
                else:
                     print(f"[DEBUG] No EXIF data dictionary found to process for non-HEIF.") # DEBUG
            
            print(f"[DEBUG] Metadata before sanitization: {metadata}") # DEBUG
            # Final sanitization pass on the whole dictionary before returning
            sanitized_metadata = _sanitize_for_bson(metadata)
            print(f"[DEBUG] Metadata after sanitization: {sanitized_metadata}") # DEBUG
            return sanitized_metadata
    except Exception as e:
        print(f"[ERROR] Exception during metadata extraction: {e}") # DEBUG
        # Sanitize error message as well
        return _sanitize_for_bson({'error': str(e), 'filename': os.path.basename(image_path)})

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

def get_location_name(latitude, longitude):
    """Performs reverse geocoding to get a location name from coordinates."""
    if latitude is None or longitude is None:
        return None
        
    try:
        # Round coordinates slightly for potentially better caching/grouping
        lat_str = f"{latitude:.4f}"
        lon_str = f"{longitude:.4f}"
        
        print(f"[DEBUG GEO] Performing reverse lookup for: ({lat_str}, {lon_str})")
        location = geolocator.reverse((lat_str, lon_str), exactly_one=True, language='en', timeout=5)
        
        if location and location.address:
            # Attempt to parse address for a user-friendly name
            address_parts = location.address.split(',')
            if len(address_parts) >= 3:
                # Try City/Region, Country
                location_name = f"{address_parts[-3].strip()}, {address_parts[-1].strip()}"
            elif len(address_parts) > 1:
                # Fallback to second to last part (often state/region), Country
                 location_name = f"{address_parts[-2].strip()}, {address_parts[-1].strip()}"
            else:
                 location_name = address_parts[-1].strip() # Fallback to Country or full address
                 
            print(f"[DEBUG GEO] Geocoding result: {location_name}")
            return location_name
        else:
            print("[DEBUG GEO] No location found by geolocator.")
            return None
            
    except (GeocoderTimedOut, GeocoderServiceError) as geo_err:
        print(f"[WARN GEO] Geocoding service error for ({lat_str}, {lon_str}): {geo_err}")
        return None # Return None on service errors
    except Exception as e:
        # Catch any other unexpected errors during geocoding
        print(f"[ERROR GEO] Unexpected geocoding error for ({lat_str}, {lon_str}): {e}")
        return None 